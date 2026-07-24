"""Scalp-engine controls: 65% confidence floor, post-close cooldown (set on
ANY close + enforced), and the ATR-aware trailing distance."""

import sys
import types
from datetime import datetime, timezone, timedelta

import pytest

if "exchange" not in sys.modules or not hasattr(sys.modules.get("exchange"), "_TEST_STUB"):
    stub = types.ModuleType("exchange")
    stub._TEST_STUB = True
    stub.get_exchange = lambda: None
    stub.get_trade_client = lambda: None
    sys.modules["exchange"] = stub

import strategy
import trade_monitor
from trade_monitor import trail_distance_price
from backtester import _simulate_exit
from test_feature_parity import _breakout_df


@pytest.fixture(autouse=True)
def _clear_recent_signals():
    strategy.recent_signals.clear()
    yield
    strategy.recent_signals.clear()


# ---------------------------------------------------------------------------
# 65% confidence floor
# ---------------------------------------------------------------------------

def test_confidence_floor_blocks_low_confidence(monkeypatch):
    monkeypatch.setattr(strategy, "get_trade_history", lambda: [])
    monkeypatch.setattr(strategy, "MIN_SIGNAL_CONFIDENCE", 65)
    # Force a would-be signal whose final confidence lands below the floor.
    monkeypatch.setattr(strategy, "USE_XGBOOST", False)
    df = _breakout_df()
    s = strategy.get_signal("BTCUSDT", df.copy(), df.copy())
    if s is not None:
        assert s["confidence"] >= 65   # anything emitted must clear the floor


def test_confidence_floor_off_allows_lower(monkeypatch):
    monkeypatch.setattr(strategy, "get_trade_history", lambda: [])
    monkeypatch.setattr(strategy, "MIN_SIGNAL_CONFIDENCE", 0)
    df = _breakout_df()
    s = strategy.get_signal("BTCUSDT", df.copy(), df.copy())
    # With the floor off the synthetic breakout should produce a signal.
    assert s is not None


# ---------------------------------------------------------------------------
# Post-close cooldown: set on ANY close, enforced by is_symbol_in_cooldown
# ---------------------------------------------------------------------------

def test_cooldown_set_on_win_and_enforced(monkeypatch, tmp_path):
    import config
    import trade_manager as tm
    import db as db_module

    # Isolated sqlite so we don't touch real state.
    monkeypatch.setattr(db_module, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setattr(db_module, "_conn", None)
    monkeypatch.setattr(tm, "STORAGE_BACKEND", "sqlite")
    monkeypatch.setattr(config, "POST_CLOSE_COOLDOWN_MINUTES", 120)

    tm.save_open_trades([{"symbol": "SOLUSDT", "status": "OPEN", "entry": 100.0,
                          "sl": 98.0, "direction": "LONG", "qty": 1.0}])
    tm.save_trade_history([])

    # A WINNING close must still start the cooldown (was loss-only before).
    tm.close_trade("SOLUSDT", 104.0, "WIN", extra_fields={"pnl": 4.0})
    assert tm.is_symbol_in_cooldown("SOLUSDT") is True
    assert tm.get_cooldown_remaining("SOLUSDT") > 100

    if db_module._conn:
        db_module._conn.close()
        db_module._conn = None


def test_cooldown_disabled_when_zero(monkeypatch, tmp_path):
    import config
    import trade_manager as tm
    import db as db_module

    monkeypatch.setattr(db_module, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "t2.db"))
    monkeypatch.setattr(db_module, "_conn", None)
    monkeypatch.setattr(tm, "STORAGE_BACKEND", "sqlite")
    monkeypatch.setattr(config, "POST_CLOSE_COOLDOWN_MINUTES", 0)

    tm.save_open_trades([{"symbol": "XRPUSDT", "status": "OPEN", "entry": 1.0,
                          "sl": 0.98, "direction": "LONG", "qty": 1.0}])
    tm.save_trade_history([])
    tm.close_trade("XRPUSDT", 0.98, "LOSS", extra_fields={"pnl": -2.0})
    assert tm.is_symbol_in_cooldown("XRPUSDT") is False

    if db_module._conn:
        db_module._conn.close()
        db_module._conn = None


# ---------------------------------------------------------------------------
# ATR-aware trailing distance
# ---------------------------------------------------------------------------

def test_trail_distance_uses_wider_of_atr_and_pct(monkeypatch):
    monkeypatch.setattr(trade_monitor, "TRAIL_PERCENT", 0.3)   # 0.3% of price
    monkeypatch.setattr(trade_monitor, "TRAIL_ATR_MULT", 1.5)
    # price 100 -> pct dist 0.30; atr 1.0 * 1.5 = 1.50 -> ATR wins (much wider).
    d = trail_distance_price({"atr": 1.0}, 100.0)
    assert d == pytest.approx(1.5)
    # Tiny ATR -> percent floor holds.
    d2 = trail_distance_price({"atr": 0.01}, 100.0)
    assert d2 == pytest.approx(0.3)


def test_trail_distance_falls_back_without_atr(monkeypatch):
    monkeypatch.setattr(trade_monitor, "TRAIL_PERCENT", 0.5)
    monkeypatch.setattr(trade_monitor, "TRAIL_ATR_MULT", 1.5)
    assert trail_distance_price({}, 200.0) == pytest.approx(1.0)  # 0.5% of 200


def test_backtester_wide_trail_lets_winner_run_further():
    # Winner ratchets up; a wide ATR trail should not stop it on the shallow
    # pullbacks a 0.3% trail would have caught.
    highs = [110, 120, 130, 130]
    lows = [109.6, 118.0, 128.0, 123.0]
    closes = [110, 119, 129, 124]
    # Tight percent trail (0.3%) stops early; wide ATR trail (distance 4.0)
    # survives the mid-run pullbacks and exits higher.
    p_tight, _, _ = _simulate_exit("LONG", 100, 97, 110, highs, lows, closes, 0.3, 0.9)
    p_wide, _, _ = _simulate_exit("LONG", 100, 97, 110, highs, lows, closes, 0.3, 0.9,
                                  trail_distance=4.0)
    assert p_wide > p_tight
