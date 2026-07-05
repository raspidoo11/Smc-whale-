"""Candle Range Theory feature: sweep-and-reclaim of the previous 1H candle's
range, resampled from the 5m frame."""

import numpy as np
import pandas as pd
import pytest

import strategy
from strategy import crt_flags
from xgboost_trainer import extract_pro_features_from_trade


def _df_5m(hours, seed=0):
    """Flat 5m frame covering `hours` full hours at price ~100."""
    n = hours * 12
    rng = np.random.default_rng(seed)
    close = 100 + rng.normal(0, 0.05, n)
    df = pd.DataFrame({
        "open": close, "close": close,
        "high": close + 0.1, "low": close - 0.1,
        "volume": np.full(n, 100.0),
    }, index=pd.date_range("2024-01-01 00:00", periods=n, freq="5min", tz="UTC"))
    return df


def test_bull_crt_sweep_and_reclaim_of_prev_hour_low():
    df = _df_5m(3)
    # Previous completed hour = 01:00-02:00. Force its low to a clear level.
    prev_hour = (df.index >= "2024-01-01 01:00") & (df.index < "2024-01-01 02:00")
    df.loc[prev_hour, "low"] = 99.5
    # Current hour (02:00+): one bar sweeps below 99.5, latest close back above.
    df.iloc[-3, df.columns.get_loc("low")] = 99.3   # the purge
    df.iloc[-1, df.columns.get_loc("close")] = 100.2  # reclaimed

    bull, bear = crt_flags(df)
    assert bull == 1 and bear == 0


def test_bear_crt_sweep_and_reclaim_of_prev_hour_high():
    df = _df_5m(3)
    prev_hour = (df.index >= "2024-01-01 01:00") & (df.index < "2024-01-01 02:00")
    df.loc[prev_hour, "high"] = 100.5
    df.iloc[-3, df.columns.get_loc("high")] = 100.8  # purge above
    df.iloc[-1, df.columns.get_loc("close")] = 99.9   # closed back inside

    bull, bear = crt_flags(df)
    assert bear == 1 and bull == 0


def test_no_crt_when_no_sweep():
    df = _df_5m(3)
    bull, bear = crt_flags(df)
    assert bull == 0 and bear == 0


def test_no_crt_when_swept_but_not_reclaimed():
    df = _df_5m(3)
    prev_hour = (df.index >= "2024-01-01 01:00") & (df.index < "2024-01-01 02:00")
    df.loc[prev_hour, "low"] = 99.5
    df.iloc[-3, df.columns.get_loc("low")] = 99.3
    df.iloc[-1, df.columns.get_loc("close")] = 99.4  # still below the level

    bull, bear = crt_flags(df)
    assert bull == 0  # swept but NOT reclaimed -> no CRT


def test_crt_features_in_featurizer():
    base = {"direction": "LONG", "entry": 100.0, "sl": 98.0, "tp": 104.0}
    f = extract_pro_features_from_trade({**base, "crt": 1, "sweep": 1})
    assert f["crt"] == 1 and f["crt_x_sweep"] == 1
    f = extract_pro_features_from_trade({**base, "crt": 1, "sweep": 0})
    assert f["crt_x_sweep"] == 0
    f = extract_pro_features_from_trade(base)  # legacy rows: default 0
    assert f["crt"] == 0 and f["crt_x_sweep"] == 0


def test_signal_snapshot_carries_crt(monkeypatch):
    from test_feature_parity import _breakout_df
    strategy.recent_signals.clear()
    monkeypatch.setattr(strategy, "get_trade_history", lambda: [])
    s = strategy.get_signal("BTCUSDT", _breakout_df().copy(), _breakout_df().copy())
    strategy.recent_signals.clear()
    assert s is not None and "crt" in s and s["crt"] in (0, 1)
