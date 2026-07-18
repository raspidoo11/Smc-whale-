"""SCAN_MODE wiring + swing risk geometry (tight zone SL, big TP)."""

import sys
import types
from datetime import datetime, timezone

import pytest

if "exchange" not in sys.modules or not hasattr(sys.modules.get("exchange"), "_TEST_STUB"):
    stub = types.ModuleType("exchange")
    stub._TEST_STUB = True
    stub.get_exchange = lambda: None
    stub.get_trade_client = lambda: None
    sys.modules["exchange"] = stub

import strategy
from strategy import zone_stop
from config import _tf_minutes
from test_feature_parity import _breakout_df


@pytest.fixture(autouse=True)
def _clear_recent_signals():
    strategy.recent_signals.clear()
    yield
    strategy.recent_signals.clear()


# ---------------------------------------------------------------------------
# Timeframe parsing + candle buckets
# ---------------------------------------------------------------------------

def test_tf_minutes_parsing():
    assert _tf_minutes("5m") == 5
    assert _tf_minutes("30m") == 30
    assert _tf_minutes("4h") == 240
    assert _tf_minutes("1d") == 1440
    with pytest.raises(ValueError):
        _tf_minutes("banana")


def test_candle_bucket_follows_entry_tf(monkeypatch):
    import main
    t = datetime(2026, 7, 18, 14, 47, 33, tzinfo=timezone.utc)

    monkeypatch.setattr(main, "ENTRY_TF_MINUTES", 5)
    b = main._current_candle_bucket(t)
    assert (b.hour, b.minute, b.second) == (14, 45, 0)

    monkeypatch.setattr(main, "ENTRY_TF_MINUTES", 30)
    b = main._current_candle_bucket(t)
    assert (b.hour, b.minute, b.second) == (14, 30, 0)


# ---------------------------------------------------------------------------
# Zone stop (swing "small SL, big TP" geometry)
# ---------------------------------------------------------------------------

def test_zone_stop_distance(monkeypatch):
    monkeypatch.setattr(strategy, "MIN_SL_ATR", 0.60)
    monkeypatch.setattr(strategy, "STRUCTURE_SL_BUFFER_ATR", 0.25)
    # LONG: 0.85 ATR below reference; SHORT mirrored above.
    assert zone_stop("LONG", 100.0, atr=2.0) == pytest.approx(100.0 - 1.7)
    assert zone_stop("SHORT", 100.0, atr=2.0) == pytest.approx(100.0 + 1.7)


def test_swing_geometry_small_sl_big_tp(monkeypatch):
    """With swing settings active, a signal's stop hugs the entry zone
    (~0.85 ATR) while TP sits RR_HIGH-tier multiples away."""
    monkeypatch.setattr(strategy, "get_trade_history", lambda: [])
    monkeypatch.setattr(strategy, "SL_STYLE", "zone")
    monkeypatch.setattr(strategy, "MIN_SL_ATR", 0.60)
    monkeypatch.setattr(strategy, "STRUCTURE_SL_BUFFER_ATR", 0.25)
    monkeypatch.setattr(strategy, "RR_LOW", 3.0)
    monkeypatch.setattr(strategy, "RR_MID", 3.5)
    monkeypatch.setattr(strategy, "RR_HIGH", 4.0)

    df = _breakout_df()
    s = strategy.get_signal("BTCUSDT", df.copy(), df.copy())
    assert s is not None and s["direction"] == "LONG"

    ref = s.get("limit_price", s["entry"])
    sl_dist = abs(ref - s["sl"])
    tp_dist = abs(s["tp"] - ref)

    # Tight stop: exactly the zone distance, not parked beyond wide structure.
    assert sl_dist == pytest.approx(0.85 * s["atr"], rel=1e-6)
    # Big target: at least the lowest swing tier.
    assert tp_dist / sl_dist >= 3.0 - 1e-9
    assert s["risk_reward"] >= 3.0 - 1e-9
