"""Soft prediction limits: zones, soft gates, structure invalidation."""

import sys
import types
from datetime import datetime, timezone

import pandas as pd
import pytest

if "exchange" not in sys.modules or not hasattr(sys.modules.get("exchange"), "_TEST_STUB"):
    stub = types.ModuleType("exchange")
    stub._TEST_STUB = True
    stub.get_exchange = lambda: None
    stub.get_trade_client = lambda: None
    sys.modules["exchange"] = stub

import strategy
from strategy import (
    choose_limit_zone,
    recent_liquidity_sweep,
    find_order_block,
    get_signal,
)
from trade_monitor import _structure_invalidated
from test_feature_parity import _breakout_df


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    strategy.recent_signals.clear()
    monkeypatch.setattr(strategy, "get_trade_history", lambda: [])
    yield
    strategy.recent_signals.clear()


def test_choose_limit_zone_long_below_close():
    idx = pd.date_range("2024-01-01", periods=20, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": [100.0] * 20, "high": [101.0] * 20, "low": [99.0] * 20,
        "close": [100.5] * 20, "volume": [100.0] * 20, "atr": [1.0] * 20,
    }, index=idx)
    limit, zone = choose_limit_zone(
        "LONG", entry=105.0, sl=100.0, atr=1.0, df_5m=df,
        bull_fvg=False, bear_fvg=False,
    )
    assert limit <= 105.0
    assert limit >= 100.0 + 0.30 * 5.0 - 1e-9
    assert zone in ("atr_pullback", "order_block", "fvg")


def test_choose_limit_zone_short_above_close():
    idx = pd.date_range("2024-01-01", periods=20, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": [100.0] * 20, "high": [101.0] * 20, "low": [99.0] * 20,
        "close": [100.0] * 20, "volume": [100.0] * 20, "atr": [1.0] * 20,
    }, index=idx)
    limit, zone = choose_limit_zone(
        "SHORT", entry=95.0, sl=100.0, atr=1.0, df_5m=df,
        bull_fvg=False, bear_fvg=False,
    )
    assert limit >= 95.0
    assert limit <= 100.0 - 0.30 * 5.0 + 1e-9
    assert zone in ("atr_pullback", "order_block", "fvg")


def test_signal_has_prediction_fields():
    s = get_signal("BTCUSDT", _breakout_df().copy(), _breakout_df().copy())
    assert s is not None
    for key in (
        "zone_type", "invalidation_price", "structure_swing",
        "setup_score", "signal_close", "limit_price",
    ):
        assert key in s, f"missing {key}"
    assert s["limit_price"] <= s["signal_close"]
    assert s["sl"] < s["limit_price"]


def test_structure_invalidation_long():
    trade = {"invalidation_price": 99.0, "sl": 98.5, "entry": 100.0}
    assert _structure_invalidated("LONG", 98.9, trade) is True
    assert _structure_invalidated("LONG", 99.5, trade) is False


def test_structure_invalidation_short():
    trade = {"invalidation_price": 101.0, "entry": 100.0}
    assert _structure_invalidated("SHORT", 101.1, trade) is True
    assert _structure_invalidated("SHORT", 100.5, trade) is False


def test_recent_sweep_detects_prior_bar():
    n = 40
    o = [100.0] * n
    h = [100.5] * n
    l = [99.5] * n
    c = [100.1] * n
    v = [100.0] * n
    # Bar 31 sweeps prior swing and closes green; later bars quiet.
    l[31] = 99.0
    o[31] = 99.6
    c[31] = 99.9
    h[31] = 100.0
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)
    assert recent_liquidity_sweep(df, "LONG", lookback=10, swing_window=12) is True


def test_find_order_block_returns_price_or_none():
    df = _breakout_df()
    df = strategy.calculate_features(df)
    ob = find_order_block(df, "LONG")
    assert ob is None or isinstance(ob, float)
