"""Covers the newly-wired AI machinery: the dynamic confidence threshold and
the expected-R entry filter."""

import numpy as np
import pandas as pd
import pytest

import strategy
from strategy import get_signal
from xgboost_trainer import get_dynamic_confidence_threshold


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    strategy.recent_signals.clear()
    monkeypatch.setattr(strategy, "get_trade_history", lambda: [])
    yield
    strategy.recent_signals.clear()


def _breakout_df(n=60):
    o, h, l, c, v = [], [], [], [], []
    for _ in range(n - 3):
        o.append(100.0); c.append(100.2); h.append(100.6); l.append(99.6); v.append(100.0)
    o += [100.2, 101.5, 103.0]
    c += [101.5, 103.0, 105.0]
    h += [101.8, 103.4, 105.3]
    l += [100.0, 101.3, 102.8]
    v += [500.0, 550.0, 600.0]
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)


# ----- dynamic threshold (pure function) -----

def test_dynamic_threshold_trending_is_looser_than_volatile():
    trending = get_dynamic_confidence_threshold(regime="trending", atr_percentile=50, recent_win_rate=0.5)
    volatile = get_dynamic_confidence_threshold(regime="volatile", atr_percentile=50, recent_win_rate=0.5)
    assert trending < volatile


def test_dynamic_threshold_tightens_after_losses():
    calm = get_dynamic_confidence_threshold(regime="ranging", atr_percentile=50, recent_win_rate=0.60)
    cold = get_dynamic_confidence_threshold(regime="ranging", atr_percentile=50, recent_win_rate=0.30)
    assert cold > calm


def test_dynamic_threshold_bounded():
    for regime in ("trending", "ranging", "volatile"):
        for wr in (0.0, 0.5, 1.0):
            for atr in (0, 50, 100):
                t = get_dynamic_confidence_threshold(regime=regime, atr_percentile=atr, recent_win_rate=wr)
                assert 30 <= t <= 75


# ----- expected-R filter (wired into get_signal, AI mode) -----

# The veto only applies once the model has learned from >=60 REAL closed
# trades (backtest-backfilled rows don't count) — see strategy.py.
_REAL_HISTORY_60 = [
    {"symbol": "XUSDT", "status": "WIN" if i % 2 else "LOSS", "pnl": 1.0 if i % 2 else -1.0}
    for i in range(60)
]


def test_expected_r_filter_rejects_low_r(monkeypatch):
    monkeypatch.setattr(strategy, "USE_XGBOOST", True)
    monkeypatch.setattr(strategy, "get_trade_history", lambda: list(_REAL_HISTORY_60))
    monkeypatch.setattr(strategy, "get_xgboost_probability", lambda f: 90.0)
    monkeypatch.setattr(strategy, "get_dynamic_confidence_threshold", lambda **k: 40)
    monkeypatch.setattr(strategy, "MIN_EXPECTED_R", 0.0)
    monkeypatch.setattr(strategy, "get_expected_r", lambda f: -0.5)  # model expects a loss in R

    assert get_signal("BTCUSDT", _breakout_df().copy(), _breakout_df().copy()) is None


def test_expected_r_filter_dormant_below_60_real_trades(monkeypatch):
    # 59 real closed trades (plus any number of backtest rows) -> the veto must
    # NOT fire, even on a terrible expected R: a warm-start model must not
    # bias the live training data it is trying to learn from.
    history = list(_REAL_HISTORY_60[:59]) + [
        {"symbol": "XUSDT", "status": "LOSS", "pnl": -1.0, "source": "backtest"}
    ] * 100
    monkeypatch.setattr(strategy, "USE_XGBOOST", True)
    monkeypatch.setattr(strategy, "get_trade_history", lambda: history)
    monkeypatch.setattr(strategy, "get_xgboost_probability", lambda f: 90.0)
    monkeypatch.setattr(strategy, "get_dynamic_confidence_threshold", lambda **k: 40)
    monkeypatch.setattr(strategy, "MIN_EXPECTED_R", 0.0)
    monkeypatch.setattr(strategy, "get_expected_r", lambda f: -0.5)

    assert get_signal("BTCUSDT", _breakout_df().copy(), _breakout_df().copy()) is not None


def test_expected_r_filter_allows_good_r(monkeypatch):
    monkeypatch.setattr(strategy, "USE_XGBOOST", True)
    monkeypatch.setattr(strategy, "get_xgboost_probability", lambda f: 90.0)
    monkeypatch.setattr(strategy, "get_dynamic_confidence_threshold", lambda **k: 40)
    monkeypatch.setattr(strategy, "MIN_EXPECTED_R", 0.0)
    monkeypatch.setattr(strategy, "get_expected_r", lambda f: 1.2)

    sig = get_signal("BTCUSDT", _breakout_df().copy(), _breakout_df().copy())
    assert sig is not None
    assert sig["expected_r"] == 1.2


def test_no_expected_r_model_does_not_filter(monkeypatch):
    # expected_r None (model not trained yet) must not block trades.
    monkeypatch.setattr(strategy, "USE_XGBOOST", True)
    monkeypatch.setattr(strategy, "get_xgboost_probability", lambda f: 90.0)
    monkeypatch.setattr(strategy, "get_dynamic_confidence_threshold", lambda **k: 40)
    monkeypatch.setattr(strategy, "get_expected_r", lambda f: None)

    sig = get_signal("BTCUSDT", _breakout_df().copy(), _breakout_df().copy())
    assert sig is not None
