"""The most important test in the suite: guard the train/serve feature
contract. A live signal must carry every raw field the trainer's featurizer
reads, so the model scores production trades on the same feature vector it was
trained on."""

import math
import numpy as np
import pandas as pd
import pytest

import strategy
from strategy import get_signal, calculate_features
from xgboost_trainer import extract_pro_features_from_trade, calculate_historical_context


@pytest.fixture(autouse=True)
def _clear_recent_signals():
    # recent_signals is module-level state that dedups by signal_hash; clear it
    # so one test's signal doesn't cooldown-suppress the next test's identical
    # synthetic candle.
    strategy.recent_signals.clear()
    yield
    strategy.recent_signals.clear()


def _breakout_df(n=60):
    """Synthetic OHLCV: a long flat base then a sharp bullish break-of-structure
    with a volume spike + displacement, engineered to reliably fire a LONG."""
    o, h, l, c, v = [], [], [], [], []
    for _ in range(n - 3):
        o.append(100.0); c.append(100.2); h.append(100.6); l.append(99.6); v.append(100.0)
    o += [100.2, 101.5, 103.0]
    c += [101.5, 103.0, 105.0]
    h += [101.8, 103.4, 105.3]
    l += [100.0, 101.3, 102.8]
    v += [500.0, 550.0, 600.0]
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx
    )


def test_signal_carries_full_feature_contract(monkeypatch):
    # Isolate from any real trade_history.json on disk.
    monkeypatch.setattr(strategy, "get_trade_history", lambda: [])

    df = _breakout_df()
    signal = get_signal("BTCUSDT", df.copy(), df.copy())

    assert signal is not None, "synthetic breakout should produce a signal"
    assert signal["direction"] == "LONG"

    # Every contextual raw field the trainer relies on must be persisted on the
    # signal (these were the dead-constant / skewed features before the fix).
    for key in (
        "market_regime", "atr_percentile", "body", "volume", "volume_ma",
        "distance_to_ema20", "distance_to_ema50", "distance_to_vwap",
        "distance_to_prev_high", "distance_to_prev_low",
    ):
        assert key in signal, f"signal missing raw feature: {key}"
        assert signal[key] is not None

    # The trainer's featurizer must consume the live signal without KeyError,
    # and every produced feature must be finite (no NaN/inf leaking into XGB).
    context = calculate_historical_context([])
    feats = extract_pro_features_from_trade(signal, context, regime=signal["market_regime"])
    for name, val in feats.items():
        if isinstance(val, (int, float, np.floating)):
            assert math.isfinite(float(val)), f"non-finite feature: {name}={val}"


def test_signal_hash_includes_symbol(monkeypatch):
    monkeypatch.setattr(strategy, "get_trade_history", lambda: [])
    df = _breakout_df()
    s1 = get_signal("BTCUSDT", df.copy(), df.copy())
    strategy.recent_signals.clear()
    s2 = get_signal("ETHUSDT", df.copy(), df.copy())

    assert s1["signal_hash"].startswith("BTCUSDT_")
    assert s2["signal_hash"].startswith("ETHUSDT_")
    # Same candle + same direction on two symbols must NOT collide.
    assert s1["signal_hash"] != s2["signal_hash"]


def test_features_include_ema_and_vwap():
    df = calculate_features(_breakout_df())
    for col in ("ema20", "ema50", "vwap"):
        assert col in df.columns
        assert pd.notna(df[col].iloc[-1])
