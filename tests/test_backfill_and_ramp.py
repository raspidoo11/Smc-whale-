"""Tests for the small-sample safeguards: the AI vote ramp, source-weighted
training rows, and the backtest-backfill transform."""

import numpy as np
import pytest

from strategy import ai_blend_weight
from xgboost_trainer import build_feature_frame, BACKTEST_SAMPLE_WEIGHT
from backfill_from_backtest import prepare_backfill


# ---------------------------------------------------------------------------
# ai_blend_weight: trust is earned with sample size
# ---------------------------------------------------------------------------

def test_ramp_zero_below_min_trades():
    assert ai_blend_weight(0) == 0.0
    assert ai_blend_weight(30) == 0.0


def test_ramp_small_at_42_trades():
    w = ai_blend_weight(42)
    assert 0.0 < w < 0.05  # ~4% say, not 40%


def test_ramp_full_at_150_and_beyond():
    assert ai_blend_weight(150) == 0.40
    assert ai_blend_weight(500) == 0.40


def test_ramp_monotonic():
    values = [ai_blend_weight(n) for n in range(0, 200, 10)]
    assert values == sorted(values)


# ---------------------------------------------------------------------------
# Source-aware training rows
# ---------------------------------------------------------------------------

def _closed(source=None, status="WIN"):
    t = {"direction": "LONG", "entry": 100.0, "sl": 98.0, "tp": 104.0,
         "exit_price": 104.0 if status == "WIN" else 98.0,
         "status": status, "pnl": 4.0 if status == "WIN" else -2.0}
    if source:
        t["source"] = source
    return t


def test_feature_frame_carries_sample_source():
    history = [_closed("backtest"), _closed("backtest", "LOSS"), _closed(), _closed(None, "LOSS")]
    df = build_feature_frame(history)
    assert list(df["sample_source"]) == ["backtest", "backtest", "live", "live"]


def test_sample_source_not_a_model_feature():
    from xgboost_trainer import prepare_X_y
    df = build_feature_frame([_closed("backtest"), _closed(None, "LOSS")])
    X, y = prepare_X_y(df)
    assert "sample_source" not in X.columns  # metadata, not signal


def test_backtest_weight_is_reduced():
    assert 0 < BACKTEST_SAMPLE_WEIGHT < 1


# ---------------------------------------------------------------------------
# prepare_backfill: tag, sort, cap
# ---------------------------------------------------------------------------

def test_prepare_backfill_tags_and_sorts():
    by_symbol = {
        "BTC/USDT:USDT": [{"entry_time": "2026-07-03 10:00", "status": "WIN"}],
        "ETH/USDT:USDT": [{"entry_time": "2026-07-01 09:00", "status": "LOSS"}],
    }
    rows = prepare_backfill(by_symbol, max_trades=10)
    assert all(r["source"] == "backtest" for r in rows)
    assert rows[0]["entry_time"] < rows[1]["entry_time"]  # chronological
    assert rows[0]["symbol"] == "ETH/USDT:USDT"


def test_prepare_backfill_caps_keeping_most_recent():
    by_symbol = {"BTC/USDT:USDT": [
        {"entry_time": f"2026-07-0{d} 10:00", "status": "WIN"} for d in range(1, 6)
    ]}
    rows = prepare_backfill(by_symbol, max_trades=2)
    assert len(rows) == 2
    assert [r["entry_time"] for r in rows] == ["2026-07-04 10:00", "2026-07-05 10:00"]
