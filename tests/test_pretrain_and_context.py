"""Tests for the offline-training upgrades: R-based labels, the historical
context provider, BOS-series parity, purged walk-forward folds, and the
Fear & Greed feature."""

import numpy as np
import pandas as pd
import pytest

import strategy
from xgboost_trainer import build_feature_frame, extract_pro_features_from_trade
from historical_context import (
    HistoricalContextProvider,
    _series_lookup,
    compute_bos_series,
)
from pretrain import purged_walk_forward_folds


# ---------------------------------------------------------------------------
# Label engineering: positive class = realized R >= 0.5
# ---------------------------------------------------------------------------

def _trade(entry, sl, exit_price, status):
    return {"direction": "LONG", "entry": entry, "sl": sl, "tp": entry * 1.04,
            "exit_price": exit_price, "status": status, "pnl": exit_price - entry}


def test_labels_use_r_threshold_not_pnl_sign():
    history = [
        _trade(100, 98, 100.2, "WIN"),   # +0.1R scratch "win"  -> label 0
        _trade(100, 98, 104.0, "WIN"),   # +2.0R runner         -> label 1
        _trade(100, 98, 98.0, "LOSS"),   # -1.0R loss           -> label 0
        _trade(100, 98, 101.0, "WIN"),   # +0.5R exactly        -> label 1
    ]
    df = build_feature_frame(history)
    assert list(df["target"]) == [0, 1, 0, 1]


# ---------------------------------------------------------------------------
# Historical series lookup + provider
# ---------------------------------------------------------------------------

def test_series_lookup_most_recent_at_or_before():
    ts = [100, 200, 300]
    vals = [1.0, 2.0, 3.0]
    assert _series_lookup(ts, vals, 99) is None    # before series starts
    assert _series_lookup(ts, vals, 100) == 1.0
    assert _series_lookup(ts, vals, 250) == 2.0
    assert _series_lookup(ts, vals, 999) == 3.0


def test_provider_lookup_uses_replayed_bar_time(monkeypatch):
    provider = HistoricalContextProvider()
    provider._funding["X"] = {"ts": [1000, 2000], "vals": [0.01, 0.02]}
    provider._oi["X"] = {"ts": [1000], "vals": [5.0]}
    provider._btc = {"ts": [1000, 2000], "vals": [1, -1]}
    provider._fng = {"ts": [1000], "vals": [25.0]}

    from datetime import datetime, timezone
    monkeypatch.setattr(
        strategy, "NOW_FN",
        lambda: datetime.fromtimestamp(1.5, tz=timezone.utc),  # 1500 ms
    )
    ctx = provider("X")
    assert ctx["funding_rate"] == 0.01
    assert ctx["oi_change_pct"] == 5.0
    assert ctx["btc_trend"] == 1
    assert ctx["fng"] == 25.0
    assert ctx["spread_pct"] is None


def test_compute_bos_series_matches_strategy_functions():
    rng = np.random.default_rng(3)
    price = 100 + np.cumsum(rng.normal(0, 0.5, 120))
    df = pd.DataFrame({
        "open": price, "close": price + rng.normal(0, 0.2, 120),
        "high": price + rng.uniform(0.2, 0.8, 120),
        "low": price - rng.uniform(0.2, 0.8, 120),
        "volume": rng.uniform(50, 150, 120),
    }, index=pd.date_range("2024-01-01", periods=120, freq="15min", tz="UTC"))

    series = compute_bos_series(df)
    # Compare the vectorized series against the live functions at several bars.
    for t in (30, 50, 80, 119):
        window = df.iloc[: t + 1]
        expected = 1 if strategy.bullish_bos(window) else (-1 if strategy.bearish_bos(window) else 0)
        assert series.iloc[t] == expected, f"BOS mismatch at bar {t}"


# ---------------------------------------------------------------------------
# Purged walk-forward folds
# ---------------------------------------------------------------------------

def test_purged_folds_respect_embargo_and_order():
    hour = 3600 * 1000
    entry_ms = [i * hour for i in range(100)]
    exit_ms = [e + 2 * hour for e in entry_ms]  # each trade lives 2 hours

    folds = list(purged_walk_forward_folds(entry_ms, exit_ms, n_folds=4, embargo_ms=5 * hour))
    assert folds, "expected at least one evaluable fold"
    for train_idx, test_idx in folds:
        test_start = entry_ms[test_idx[0]]
        assert max(train_idx) < min(test_idx)  # strictly walk-forward
        for i in train_idx:
            assert exit_ms[i] <= test_start - 5 * hour  # purge + embargo held


# ---------------------------------------------------------------------------
# Fear & Greed featurization
# ---------------------------------------------------------------------------

def test_fng_features():
    base = {"direction": "LONG", "entry": 100.0, "sl": 98.0, "tp": 104.0}
    f = extract_pro_features_from_trade({**base, "fng": 12})
    assert f["fng"] == 12 and f["is_extreme_fear"] == 1 and f["is_extreme_greed"] == 0
    f = extract_pro_features_from_trade({**base, "fng": 91})
    assert f["is_extreme_greed"] == 1 and f["is_extreme_fear"] == 0
    f = extract_pro_features_from_trade(base)  # missing -> neutral 50
    assert f["fng"] == 50.0 and f["is_extreme_fear"] == 0 and f["is_extreme_greed"] == 0
