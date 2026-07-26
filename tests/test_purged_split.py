"""Chronological/purged split + EV-based promotion gate.

These cover the two defects that made every previous promotion decision
untrustworthy:

  1. `split_train_holdout` shuffled the time series, so holdout metrics scored
     a model on periods it had trained across.
  2. `promotion_decision`'s "no existing champion" path auto-promoted, which is
     how a model with 0.357 holdout AUC (worse than chance) reached production.
"""

import numpy as np
import pandas as pd
import pytest

from xgboost_trainer import (
    META_COLS,
    MIN_TRAIN_ROWS,
    build_feature_frame,
    prepare_X_y,
    promotion_decision,
    split_train_holdout,
    trade_entry_ms,
    trade_exit_ms,
)

HOUR_MS = 3600 * 1000


def make_frame(targets, spacing_hours=48, hold_hours=1):
    """Chronological frame with one row per target, spaced far enough apart
    that the default 24h embargo purges nothing."""
    rows = []
    for i, t in enumerate(targets):
        entry = i * spacing_hours * HOUR_MS
        rows.append({
            "feat": float(i),
            "target": int(t),
            "realized_r": 1.0 if t else -1.0,
            "_entry_ms": entry,
            "_exit_ms": entry + hold_hours * HOUR_MS,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Chronology
# --------------------------------------------------------------------------

def test_split_is_chronological_no_future_rows_in_train():
    df = make_frame([i % 2 for i in range(40)])
    train, test = split_train_holdout(df, holdout_size=8)

    assert train is not None
    # Every training row must precede every holdout row. Under the old
    # shuffled split this failed by construction.
    assert train["_entry_ms"].max() < test["_entry_ms"].min()
    assert set(train.index).isdisjoint(test.index)


def test_holdout_is_the_most_recent_block():
    df = make_frame([i % 2 for i in range(40)])
    _, test = split_train_holdout(df, holdout_size=8)
    assert list(test.index) == list(range(32, 40))


# --------------------------------------------------------------------------
# Purge + embargo
# --------------------------------------------------------------------------

def test_purge_drops_trades_still_open_when_holdout_starts():
    # Spaced 24h, each held 48h -> the last couple of training trades are still
    # unresolved (or inside the embargo) when the holdout opens, and must go.
    df = make_frame([i % 2 for i in range(60)], spacing_hours=24, hold_hours=48)
    train, test = split_train_holdout(df, holdout_size=10)

    assert train is not None
    cutoff = test["_entry_ms"].min() - 24 * HOUR_MS
    # Nothing kept may have resolved after the embargoed cutoff.
    assert (train["_exit_ms"] <= cutoff).all()
    # And the purge must actually have removed something.
    assert len(train) < 50, "expected the trailing overlapping trades to be purged"


def test_everything_purged_yields_no_split_rather_than_a_leaky_one():
    # Trades held 500h while spaced 1h apart: no training trade has resolved by
    # the time the holdout opens, so there is no honest split at all.
    df = make_frame([i % 2 for i in range(60)], spacing_hours=1, hold_hours=500)
    train, test = split_train_holdout(df, holdout_size=10)
    assert train is None and test is None


def test_embargo_gap_is_enforced():
    df = make_frame([i % 2 for i in range(40)], spacing_hours=48, hold_hours=1)
    train, test = split_train_holdout(df, holdout_size=8, embargo_ms=100 * HOUR_MS)
    assert train is not None
    assert (train["_exit_ms"] <= test["_entry_ms"].min() - 100 * HOUR_MS).all()


# --------------------------------------------------------------------------
# The streak bug that shuffling was introduced to fix
# --------------------------------------------------------------------------

def test_holdout_grows_when_most_recent_block_is_single_class():
    # Last 8 trades all LOSS: the requested holdout is single-class, so the
    # split must widen backwards rather than shuffle or give up.
    targets = [i % 2 for i in range(32)] + [0] * 8
    df = make_frame(targets)
    train, test = split_train_holdout(df, holdout_size=8)

    assert train is not None, "should grow the holdout instead of failing"
    assert len(test) > 8
    assert test["target"].nunique() == 2
    assert train["target"].nunique() == 2
    # Chronology still intact after growing.
    assert train["_entry_ms"].max() < test["_entry_ms"].min()


def test_returns_none_rather_than_shuffling_when_no_honest_split_exists():
    # Single-class everywhere: no chronological split can ever be scored.
    df = make_frame([1] * 40)
    train, test = split_train_holdout(df, holdout_size=8)
    assert train is None and test is None


def test_returns_none_when_training_side_too_small():
    df = make_frame([i % 2 for i in range(MIN_TRAIN_ROWS)])
    train, _ = split_train_holdout(df, holdout_size=MIN_TRAIN_ROWS)
    assert train is None


# --------------------------------------------------------------------------
# Metadata must never reach the model
# --------------------------------------------------------------------------

def test_timestamps_are_never_used_as_features():
    df = make_frame([i % 2 for i in range(30)])
    X, y = prepare_X_y(df)

    # Raw epoch timestamps are numeric, so without an explicit drop
    # select_dtypes would train on them and learn "this date range won".
    for col in ("_entry_ms", "_exit_ms", "realized_r", "target"):
        assert col not in X.columns
    assert "feat" in X.columns
    assert len(y) == len(df)


def test_build_feature_frame_carries_timestamps_but_not_as_features():
    history = [
        {
            "status": "WIN" if i % 2 else "LOSS",
            "direction": "LONG",
            "entry": 100.0, "sl": 99.0, "tp": 102.0,
            "exit_price": 102.0 if i % 2 else 99.0,
            "entry_time": f"2026-01-{i + 1:02d}T00:00:00+00:00",
            "exit_time": f"2026-01-{i + 1:02d}T04:00:00+00:00",
            "atr": 1.0,
        }
        for i in range(6)
    ]
    df = build_feature_frame(history)

    assert df["_entry_ms"].notna().all()
    assert df["_exit_ms"].notna().all()
    assert (df["_exit_ms"] > df["_entry_ms"]).all()

    X, _ = prepare_X_y(df)
    assert not set(META_COLS) & set(X.columns)


# --------------------------------------------------------------------------
# Timestamp helpers
# --------------------------------------------------------------------------

def test_exit_ms_prefers_recorded_exit_time():
    trade = {
        "entry_time": "2026-01-01T00:00:00+00:00",
        "exit_time": "2026-01-01T06:00:00+00:00",
        "bars_held": 999,
    }
    assert trade_exit_ms(trade) - trade_entry_ms(trade) == 6 * HOUR_MS


def test_exit_ms_falls_back_to_bars_held():
    from config import ENTRY_TF_MINUTES
    trade = {"entry_time": "2026-01-01T00:00:00+00:00", "bars_held": 12}
    span = trade_exit_ms(trade) - trade_entry_ms(trade)
    assert span == 12 * ENTRY_TF_MINUTES * 60 * 1000


@pytest.mark.parametrize("trade", [{}, {"entry_time": None}, {"entry_time": "junk"}])
def test_timestamp_helpers_return_none_on_bad_input(trade):
    assert trade_entry_ms(trade) is None
    assert trade_exit_ms(trade) is None


# --------------------------------------------------------------------------
# Promotion gate
# --------------------------------------------------------------------------

def metrics(**kw):
    base = {"auc": 0.6, "ev_r": None, "precision_lift": None,
            "precision": 0.5, "base_rate": 0.5, "n_selected": 10}
    base.update(kw)
    return base


def test_no_champion_still_requires_clearing_an_absolute_bar():
    """Regression: a model with 0.357 AUC was promoted because there was no
    champion to compare against."""
    promote, reason = promotion_decision(metrics(auc=0.357), None)
    assert promote is False
    assert "0.357" in reason


def test_no_champion_rejects_negative_expected_r():
    promote, reason = promotion_decision(metrics(ev_r=-0.4), None)
    assert promote is False
    assert "expected R" in reason


def test_no_champion_promotes_a_model_that_clears_the_bar():
    promote, reason = promotion_decision(metrics(ev_r=0.35), None)
    assert promote is True
    assert "no existing champion" in reason


def test_expected_r_outranks_auc_in_head_to_head():
    # Challenger has WORSE AUC but better expected R on the trades it takes —
    # it should win, because EV is what pays.
    challenger = metrics(auc=0.52, ev_r=0.50)
    champion = metrics(auc=0.71, ev_r=0.10)
    promote, reason = promotion_decision(challenger, champion)
    assert promote is True
    assert "expected R" in reason


def test_challenger_must_beat_champion_by_the_margin():
    challenger = metrics(ev_r=0.101)
    champion = metrics(ev_r=0.100)
    promote, _ = promotion_decision(challenger, champion)
    assert promote is False, "a hair better is noise, not an improvement"


def test_precision_lift_used_when_expected_r_unavailable():
    promote, reason = promotion_decision(
        metrics(precision_lift=0.20), metrics(precision_lift=0.05)
    )
    assert promote is True
    assert "precision lift" in reason


def test_unevaluable_challenger_never_replaces_champion():
    promote, reason = promotion_decision(None, metrics(ev_r=0.2))
    assert promote is False
    assert "not evaluable" in reason


def test_model_that_takes_no_trades_is_not_promoted_on_ev():
    # n_selected == 0 -> evaluate_model leaves ev_r/precision None; the gate
    # must fall through to AUC rather than treating "no trades" as skill.
    promote, _ = promotion_decision(metrics(auc=0.8, ev_r=None), None)
    assert promote is True  # AUC fallback
    promote, _ = promotion_decision(metrics(auc=0.45, ev_r=None), None)
    assert promote is False
