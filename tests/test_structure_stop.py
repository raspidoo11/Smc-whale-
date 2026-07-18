"""Anti stop-hunt SL placement.

Guards the fix for the classic bait: stops sitting on the swing with a 0.05%
nudge, or sub-1.0 ATR noise room that equal-high/low sweeps farm.

Also guards HTF structure: entry-TF (5m) alone parks SL under equal-lows that
print as stop-hunt bait on the bias TF (15m chart).
"""

import numpy as np
import pandas as pd
import pytest

import strategy
from strategy import compute_structure_stop, compute_structure_stop_htf, get_signal
from config import MIN_SL_ATR, STRUCTURE_SL_BUFFER_ATR
from test_feature_parity import _breakout_df


@pytest.fixture(autouse=True)
def _clear_recent_signals():
    strategy.recent_signals.clear()
    yield
    strategy.recent_signals.clear()


def _ohlcv_with_swing(n=40, swing_low=98.0, swing_high=102.0, last_close=100.0):
    """Flat range with a clear swing low/high in the middle of the window."""
    o = [100.0] * n
    h = [100.5] * n
    l = [99.5] * n
    c = [100.0] * n
    v = [100.0] * n
    # Plant structure a few bars back (not on the last bar).
    mid = n // 2
    l[mid] = swing_low
    h[mid] = swing_high
    c[-1] = last_close
    o[-1] = last_close
    h[-1] = last_close + 0.3
    l[-1] = last_close - 0.3
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx
    )


def test_long_stop_beyond_swing_plus_buffer():
    atr = 1.0
    entry = 100.0
    df = _ohlcv_with_swing(swing_low=99.0)
    sl, swing = compute_structure_stop("LONG", df, entry, atr)
    assert swing == pytest.approx(99.0)
    # Must clear the swing by at least the buffer (structure side).
    assert sl <= swing - STRUCTURE_SL_BUFFER_ATR * atr + 1e-9
    # And never tighter than the ATR floor.
    assert sl <= entry - MIN_SL_ATR * atr + 1e-9


def test_long_stop_uses_atr_floor_when_swing_is_close():
    """Swing only 0.3 ATR under entry → floor forces wider stop."""
    atr = 2.0
    entry = 100.0
    # swing_low at 99.4 → structure_sl = 99.4 - 0.25*2 = 98.9
    # atr_floor = 100 - 1.15*2 = 97.7  → min picks floor
    df = _ohlcv_with_swing(swing_low=99.4)
    sl, swing = compute_structure_stop("LONG", df, entry, atr)
    assert sl == pytest.approx(entry - MIN_SL_ATR * atr)
    assert sl < swing  # still beyond the swing


def test_long_stop_uses_structure_when_swing_is_far():
    atr = 1.0
    entry = 100.0
    # Far swing: structure_sl = 97.0 - 0.25 = 96.75, floor = 98.85 → structure wins
    df = _ohlcv_with_swing(swing_low=97.0)
    sl, swing = compute_structure_stop("LONG", df, entry, atr)
    assert swing == pytest.approx(97.0)
    assert sl == pytest.approx(97.0 - STRUCTURE_SL_BUFFER_ATR * atr)
    assert sl < entry - MIN_SL_ATR * atr  # wider than floor


def test_short_stop_beyond_swing_plus_buffer():
    atr = 1.0
    entry = 100.0
    df = _ohlcv_with_swing(swing_high=101.0)
    sl, swing = compute_structure_stop("SHORT", df, entry, atr)
    assert swing == pytest.approx(101.0)
    assert sl >= swing + STRUCTURE_SL_BUFFER_ATR * atr - 1e-9
    assert sl >= entry + MIN_SL_ATR * atr - 1e-9


def test_short_stop_uses_atr_floor_when_swing_is_close():
    atr = 2.0
    entry = 100.0
    df = _ohlcv_with_swing(swing_high=100.6)
    sl, _ = compute_structure_stop("SHORT", df, entry, atr)
    assert sl == pytest.approx(entry + MIN_SL_ATR * atr)


def test_live_signal_respects_min_sl_atr(monkeypatch):
    monkeypatch.setattr(strategy, "get_trade_history", lambda: [])
    df = _breakout_df()
    s = get_signal("BTCUSDT", df.copy(), df.copy())
    assert s is not None and s["direction"] == "LONG"
    # Risk is measured from the resting limit (where we actually fill).
    trade_entry = s["limit_price"]
    room = abs(trade_entry - s["sl"])
    assert room >= MIN_SL_ATR * s["atr"] * 0.85 - 1e-9
    assert "structure_swing" in s
    # Must not sit *on* the swing (old 0.9995 bait).
    assert s["sl"] < s["structure_swing"]
    # Limit still on the correct side of signal close / SL.
    assert s["limit_price"] <= s["entry"]
    assert s["limit_price"] > s["sl"]
    assert "zone_type" in s and "invalidation_price" in s


def test_rejects_bad_direction():
    df = _ohlcv_with_swing()
    with pytest.raises(ValueError):
        compute_structure_stop("SIDEWAYS", df, 100.0, 1.0)


def test_htf_stop_clears_bias_swing_not_just_entry_noise():
    """5m swing near entry is bait on 15m; bias swing is deeper — SL must clear it."""
    atr = 1.0
    entry = 100.0
    # Entry TF: shallow equal-low bait just under price.
    df_5m = _ohlcv_with_swing(n=40, swing_low=99.2, last_close=100.0)
    # Bias TF: real structure well below (what you see on 15m).
    df_15m = _ohlcv_with_swing(n=40, swing_low=97.0, last_close=100.0)

    sl_entry_only, _ = compute_structure_stop("LONG", df_5m, entry, atr)
    sl, swing = compute_structure_stop_htf("LONG", df_5m, df_15m, entry, atr)

    assert swing == pytest.approx(97.0)
    assert sl <= 97.0 - STRUCTURE_SL_BUFFER_ATR * atr + 1e-9
    # Must be at least as wide as entry-only (never *tighter* than 5m).
    assert sl <= sl_entry_only + 1e-9
    # And clearly past the 5m bait level.
    assert sl < 99.2


def test_htf_stop_short_clears_bias_swing():
    atr = 1.0
    entry = 100.0
    df_5m = _ohlcv_with_swing(n=40, swing_high=100.8, last_close=100.0)
    df_15m = _ohlcv_with_swing(n=40, swing_high=103.0, last_close=100.0)

    sl_entry_only, _ = compute_structure_stop("SHORT", df_5m, entry, atr)
    sl, swing = compute_structure_stop_htf("SHORT", df_5m, df_15m, entry, atr)

    assert swing == pytest.approx(103.0)
    assert sl >= 103.0 + STRUCTURE_SL_BUFFER_ATR * atr - 1e-9
    assert sl >= sl_entry_only - 1e-9
    assert sl > 100.8


def test_htf_falls_back_when_bias_missing():
    atr = 1.0
    entry = 100.0
    df_5m = _ohlcv_with_swing(swing_low=98.5)
    sl_e, swing_e = compute_structure_stop("LONG", df_5m, entry, atr)
    sl, swing = compute_structure_stop_htf("LONG", df_5m, None, entry, atr)
    assert sl == pytest.approx(sl_e)
    assert swing == pytest.approx(swing_e)
