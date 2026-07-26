"""Triple-barrier labeling.

The label must depend only on the forward price path under a fixed exit policy —
never on trailing/time-stop management — and must resolve ambiguous candles
PESSIMISTICALLY. Optimism on an ambiguous candle is how a backtest manufactures
edge that does not survive live.
"""

import pytest

from labeling import (
    BARRIER_PROFIT,
    BARRIER_STOP,
    BARRIER_VERTICAL,
    LABEL_PROFIT,
    LABEL_STOP,
    LABEL_TIMEOUT,
    apply_triple_barrier,
    barriers_from_atr,
    label_trade,
)


def bars(*triples):
    """(high, low, close) tuples -> the three aligned sequences."""
    highs = [t[0] for t in triples]
    lows = [t[1] for t in triples]
    closes = [t[2] for t in triples]
    return highs, lows, closes


# --------------------------------------------------------------------------
# LONG
# --------------------------------------------------------------------------

def test_long_profit_hit_first():
    h, l, c = bars((101, 100, 100), (105, 101, 104))
    out = apply_triple_barrier("LONG", 100.0, 104.0, 98.0, h, l, c, max_bars=10)
    assert out.label == LABEL_PROFIT
    assert out.barrier == BARRIER_PROFIT
    assert out.bars_to_touch == 2
    assert out.realized_r == pytest.approx(2.0)   # +4 on 2 risk
    assert out.ambiguous is False


def test_long_stop_hit_first():
    h, l, c = bars((101, 100, 100), (101, 97, 98))
    out = apply_triple_barrier("LONG", 100.0, 104.0, 98.0, h, l, c, max_bars=10)
    assert out.label == LABEL_STOP
    assert out.realized_r == pytest.approx(-1.0)
    assert out.bars_to_touch == 2


def test_long_vertical_barrier_marks_to_market():
    # Never reaches either barrier; vertical barrier at 2 bars.
    h, l, c = bars((101, 99.5, 101), (102, 100, 101), (103.9, 100, 103.9))
    out = apply_triple_barrier("LONG", 100.0, 104.0, 98.0, h, l, c, max_bars=2)
    assert out.label == LABEL_TIMEOUT
    assert out.barrier == BARRIER_VERTICAL
    assert out.bars_to_touch == 2
    assert out.realized_r == pytest.approx(0.5)   # closed at 101, risk 2


# --------------------------------------------------------------------------
# SHORT (profit is BELOW entry, stop ABOVE)
# --------------------------------------------------------------------------

def test_short_profit_hit_first():
    h, l, c = bars((100, 99, 99), (99, 95, 96))
    out = apply_triple_barrier("SHORT", 100.0, 96.0, 102.0, h, l, c, max_bars=10)
    assert out.label == LABEL_PROFIT
    assert out.realized_r == pytest.approx(2.0)   # +4 on 2 risk


def test_short_stop_hit_first():
    h, l, c = bars((103, 100, 102))
    out = apply_triple_barrier("SHORT", 100.0, 96.0, 102.0, h, l, c, max_bars=10)
    assert out.label == LABEL_STOP
    assert out.realized_r == pytest.approx(-1.0)


def test_short_vertical_barrier_sign_is_correct():
    # Price drifted DOWN to 99 -> a short is in profit -> positive R.
    h, l, c = bars((100, 98.5, 99))
    out = apply_triple_barrier("SHORT", 100.0, 96.0, 102.0, h, l, c, max_bars=1)
    assert out.label == LABEL_TIMEOUT
    assert out.realized_r == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Intrabar ambiguity — the correctness-critical case
# --------------------------------------------------------------------------

def test_ambiguous_candle_resolves_as_stop_not_profit():
    """One candle spans BOTH barriers. OHLC cannot say which came first, so the
    outcome must be the adverse one and flagged ambiguous."""
    h, l, c = bars((105, 97, 100))   # touches tp 104 AND sl 98
    out = apply_triple_barrier("LONG", 100.0, 104.0, 98.0, h, l, c, max_bars=10)
    assert out.label == LABEL_STOP
    assert out.ambiguous is True
    assert out.realized_r == pytest.approx(-1.0)


def test_ambiguous_candle_short_side():
    h, l, c = bars((103, 95, 100))   # touches tp 96 AND sl 102
    out = apply_triple_barrier("SHORT", 100.0, 96.0, 102.0, h, l, c, max_bars=10)
    assert out.label == LABEL_STOP
    assert out.ambiguous is True


def test_unambiguous_candles_are_not_flagged():
    h, l, c = bars((105, 100, 104))
    out = apply_triple_barrier("LONG", 100.0, 104.0, 98.0, h, l, c, max_bars=10)
    assert out.ambiguous is False


# --------------------------------------------------------------------------
# Label is independent of exit management
# --------------------------------------------------------------------------

def test_label_ignores_what_the_trade_manager_did():
    """Same forward path labeled identically regardless of the managed exit —
    the whole point of triple-barrier labeling."""
    h, l, c = bars((101, 100, 100), (105, 101, 104))
    trade_trailed_out_early = {
        "direction": "LONG", "entry": 100.0, "tp": 104.0, "sl": 98.0,
        "exit_price": 100.5, "exit_reason": "Trailing Stop Hit",
    }
    trade_ran_to_tp = {
        "direction": "LONG", "entry": 100.0, "tp": 104.0, "sl": 98.0,
        "exit_price": 104.0, "exit_reason": "Take Profit",
    }
    a = label_trade(trade_trailed_out_early, h, l, c, max_bars=10)
    b = label_trade(trade_ran_to_tp, h, l, c, max_bars=10)
    assert a.label == b.label == LABEL_PROFIT
    assert a.realized_r == b.realized_r


# --------------------------------------------------------------------------
# Vertical barrier bounds the walk
# --------------------------------------------------------------------------

def test_max_bars_caps_the_walk():
    # Profit is reached on bar 5, but the vertical barrier closes at bar 3.
    h, l, c = bars(*[(101, 99, 100)] * 4 + [(105, 100, 104)])
    out = apply_triple_barrier("LONG", 100.0, 104.0, 98.0, h, l, c, max_bars=3)
    assert out.label == LABEL_TIMEOUT
    assert out.bars_to_touch == 3


def test_short_forward_data_ends_at_available_bars():
    h, l, c = bars((101, 99, 100))
    out = apply_triple_barrier("LONG", 100.0, 104.0, 98.0, h, l, c, max_bars=50)
    assert out.label == LABEL_TIMEOUT
    assert out.bars_to_touch == 1


def test_no_forward_data_is_not_a_breakeven_timeout():
    out = apply_triple_barrier("LONG", 100.0, 104.0, 98.0, [], [], [], max_bars=10)
    assert out.bars_to_touch == 0
    assert out.realized_r == 0.0


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

def test_degenerate_stop_raises():
    with pytest.raises(ValueError, match="degenerate stop"):
        apply_triple_barrier("LONG", 100.0, 104.0, 100.0, [101], [99], [100], max_bars=5)


def test_bad_direction_raises():
    with pytest.raises(ValueError, match="LONG or SHORT"):
        apply_triple_barrier("UP", 100.0, 104.0, 98.0, [101], [99], [100], max_bars=5)


def test_zero_max_bars_raises():
    with pytest.raises(ValueError, match="max_bars"):
        apply_triple_barrier("LONG", 100.0, 104.0, 98.0, [101], [99], [100], max_bars=0)


def test_label_trade_returns_none_on_unlabelable_row():
    h, l, c = bars((101, 99, 100))
    assert label_trade({"direction": "LONG"}, h, l, c, max_bars=5) is None
    # Degenerate geometry must be skipped, not raise.
    bad = {"direction": "LONG", "entry": 100.0, "tp": 104.0, "sl": 100.0}
    assert label_trade(bad, h, l, c, max_bars=5) is None


# --------------------------------------------------------------------------
# ATR-scaled barriers
# --------------------------------------------------------------------------

def test_barriers_from_atr_long_and_short():
    up, dn = barriers_from_atr("LONG", 100.0, 2.0, profit_mult=2.0, stop_mult=1.0)
    assert (up, dn) == (104.0, 98.0)
    up, dn = barriers_from_atr("SHORT", 100.0, 2.0, profit_mult=2.0, stop_mult=1.0)
    assert (up, dn) == (96.0, 102.0)


def test_barriers_from_atr_rejects_nonpositive_atr():
    with pytest.raises(ValueError):
        barriers_from_atr("LONG", 100.0, 0.0, 2.0, 1.0)


def test_label_trade_atr_override_ignores_sl_tp():
    h, l, c = bars((103.5, 100, 103))
    trade = {"direction": "LONG", "entry": 100.0, "tp": 999.0, "sl": 1.0, "atr": 1.0}
    # ATR barriers: tp=103, sl=99 -> profit touched on bar 1.
    out = label_trade(trade, h, l, c, max_bars=10, profit_atr=3.0, stop_atr=1.0)
    assert out.label == LABEL_PROFIT
    assert out.realized_r == pytest.approx(3.0)


# --------------------------------------------------------------------------
# Trainer integration
# --------------------------------------------------------------------------

def test_trainer_prefers_triple_barrier_r_over_managed_exit():
    from xgboost_trainer import resolve_label_r
    trade = {
        "direction": "LONG", "entry": 100.0, "sl": 98.0,
        "exit_price": 100.2,      # managed exit scratched near breakeven
        "tb_realized_r": 2.0,     # but the path actually reached +2R
    }
    r, kind = resolve_label_r(trade)
    assert kind == "triple_barrier"
    assert r == pytest.approx(2.0)


def test_trainer_falls_back_to_realized_r_on_legacy_rows():
    from xgboost_trainer import resolve_label_r
    trade = {"direction": "LONG", "entry": 100.0, "sl": 98.0, "exit_price": 104.0}
    r, kind = resolve_label_r(trade)
    assert kind == "realized"
    assert r == pytest.approx(2.0)


def test_purge_uses_the_label_horizon_not_the_managed_exit():
    from xgboost_trainer import label_horizon_ms, trade_exit_ms
    trade = {
        "entry_time": "2026-01-01T00:00:00+00:00",
        "exit_time": "2026-01-01T01:00:00+00:00",    # managed exit: 1h
        "tb_exit_time": "2026-01-01T05:00:00+00:00",  # label resolved at 5h
    }
    # The label stays unresolved until 5h, so purging must use that.
    assert label_horizon_ms(trade) > trade_exit_ms(trade)


def test_label_kind_is_metadata_not_a_feature():
    from xgboost_trainer import META_COLS
    assert "label_kind" in META_COLS
