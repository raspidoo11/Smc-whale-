import numpy as np
import pandas as pd

import strategy
from backtester import _simulate_exit, compute_metrics, simulate


def test_simulate_exit_long_stop_loss():
    # First candle already breaks below SL -> immediate stop.
    highs = [101, 102]; lows = [97, 96]; closes = [98, 97]
    price, reason, bars = _simulate_exit("LONG", 100, 98, 110, highs, lows, closes, 0.5, 0.97)
    assert price == 98 and reason == "Stop Loss Hit" and bars == 1


def test_simulate_exit_long_tp_then_trailing():
    # Candle 0 tags the activation zone (arms trailing at high=111); retraces hard.
    highs = [111, 111]; lows = [104, 100]; closes = [110, 101]
    price, reason, bars = _simulate_exit("LONG", 100, 100, 110, highs, lows, closes, 0.5, 0.97)
    assert reason == "Trailing Stop Hit"
    assert price == 111 * (1 - 0.5 / 100)


def test_simulate_exit_trailing_lets_winner_run_past_tp():
    # entry 100, tp 110 -> activation at 109.7. Trailing arms, price keeps
    # climbing (lows stay above the 0.5% trail) to ~130, then the last candle
    # retraces below the trail. Exit must be ABOVE tp (winner ran past TP).
    highs = [110, 120, 130, 130]
    lows = [109.6, 119.5, 129.5, 120]
    closes = [110, 119, 129, 121]
    price, reason, bars = _simulate_exit("LONG", 100, 100, 110, highs, lows, closes, 0.5, 0.97)
    assert reason == "Trailing Stop Hit"
    assert price > 110  # ran past the original TP instead of being capped
    assert price == 130 * (1 - 0.5 / 100)


def test_simulate_exit_short_stop_loss():
    highs = [104, 105]; lows = [99, 98]; closes = [103, 104]
    price, reason, bars = _simulate_exit("SHORT", 100, 103, 90, highs, lows, closes, 0.5, 0.97)
    assert price == 103 and reason == "Stop Loss Hit"


def test_compute_metrics_basic():
    trades = [
        {"pnl": 10, "realized_r": 2.0, "bars_held": 3},
        {"pnl": -5, "realized_r": -1.0, "bars_held": 2},
        {"pnl": 15, "realized_r": 2.5, "bars_held": 4},
    ]
    equity = [100, 110, 105, 120]
    m = compute_metrics(trades, equity)
    assert m["n_trades"] == 3
    assert m["win_rate"] == round(2 / 3, 3)
    assert m["profit_factor"] == round(25 / 5, 2)
    assert m["final_equity"] == 120.0


def _noise_df(n=220, seed=1):
    rng = np.random.default_rng(seed)
    price = 100 + np.cumsum(rng.normal(0, 0.3, n))
    high = price + rng.uniform(0.1, 0.5, n)
    low = price - rng.uniform(0.1, 0.5, n)
    vol = rng.uniform(80, 200, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": price, "high": high, "low": low, "close": price, "volume": vol}, index=idx)


def test_simulate_runs_and_restores_globals():
    orig_now = strategy.NOW_FN
    orig_xgb = strategy.USE_XGBOOST
    orig_hist = strategy.get_trade_history

    df = _noise_df()
    trades, metrics = simulate("TEST/USDT:USDT", df, df, use_xgboost=False)

    assert isinstance(trades, list)
    assert "n_trades" in metrics
    # The seams must be restored no matter what happened inside.
    assert strategy.NOW_FN is orig_now
    assert strategy.USE_XGBOOST is orig_xgb
    assert strategy.get_trade_history is orig_hist
