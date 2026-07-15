import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timezone
from trade_manager import get_trade_history
from xgboost_trainer import (
    calculate_historical_context,
    get_xgboost_probability,
    get_expected_r,
    get_dynamic_confidence_threshold,
    detect_market_regime,
    calculate_atr_percentile,
    extract_pro_features_from_trade,
)
from config import (
    MIN_EXPECTED_R,
    CONFIDENCE_REQUIRED_SMC,
    CONFIDENCE_REQUIRED_LIMIT,
    LIMIT_MIN_SETUP_SCORE,
    RETRACE_ATR_FRACTION,
    AI_MAX_WEIGHT,
    AI_WEIGHT_FULL_AT,
    MIN_SL_ATR,
    STRUCTURE_SL_BUFFER_ATR,
    STRUCTURE_SWING_LOOKBACK,
    ENTRY_MODE,
)

logger = logging.getLogger(__name__)

USE_XGBOOST = os.getenv("USE_XGBOOST", "false").lower() == "true"

# Time source seam. Live trading uses wall-clock UTC; the backtester overrides
# this to the candle's timestamp so session/hour features reflect the bar being
# replayed, not "now". Keeps a single get_signal() for both paths.
NOW_FN = lambda: datetime.now(timezone.utc)

# Market-context seam. main.py wires this to market_context.get_market_context
# at startup so live signals are enriched with funding rate / open-interest
# change / BTC trend / spread. The backtester leaves it as-is (network-free),
# and the featurizer treats the missing values as neutral defaults.
MARKET_CONTEXT_FN = lambda symbol: {}


def ai_blend_weight(n_real_closed, max_weight=None, min_trades=30, full_at=None):
    """How much say the model gets in final confidence, as a function of how
    many REAL closed trades it has learned from (backtest-backfilled rows
    don't count — they're warm-start data, not evidence).

    A model trained on 42 trades shouldn't carry the same vote as one trained
    on 300: below `min_trades` it gets zero say (pure SMC score); influence
    then ramps linearly, reaching AI_MAX_WEIGHT (env-tunable, default 0.40 —
    set 0.70 to let a proven model dominate) at AI_WEIGHT_FULL_AT trades."""
    max_weight = AI_MAX_WEIGHT if max_weight is None else max_weight
    full_at = AI_WEIGHT_FULL_AT if full_at is None else full_at
    if n_real_closed <= min_trades:
        return 0.0
    ramp = min(1.0, (n_real_closed - min_trades) / (full_at - min_trades))
    return round(max_weight * ramp, 4)

# Module-level set to track recent signals for cooldown (prevents duplicate alerts on same candle)
# For multi-pair scanners, manage recent_signals per symbol (e.g. dict of sets) or include symbol in signal_hash
recent_signals = set()


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def calculate_features(df):

    df = df.copy()

    # ATR
    df["atr"] = (
        df["high"] - df["low"]
    ).rolling(14).mean()

    # Volume MA
    df["volume_ma"] = (
        df["volume"]
    ).rolling(20).mean()

    # Trend references (used for distance-to-mean features). Computed here so
    # both live inference and stored-trade featurization see identical values.
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    df["vwap"] = (typical * df["volume"]).cumsum() / cum_vol

    # ------------------------------------
    # Adaptive Volume Multiplier (improved)
    # Uses recent ATR to decide strictness:
    # - When ATR is elevated (> avg * 1.3): lower multiplier (easier spike detection in high vol)
    # - Otherwise: higher multiplier
    # AI mode keeps overall stricter bias than SMC mode
    # ------------------------------------

    atr_vals = df["atr"].dropna()
    if len(atr_vals) >= 5:
        atr_average_local = atr_vals.tail(min(30, len(atr_vals))).mean()
        current_atr = df["atr"].iloc[-1] if len(df) > 0 else 0
        if pd.notna(current_atr) and pd.notna(atr_average_local) and atr_average_local > 0:
            if current_atr > atr_average_local * 1.3:
                volume_multiplier = 1.15 if not USE_XGBOOST else 1.35
            else:
                volume_multiplier = 1.30 if not USE_XGBOOST else 1.50
        else:
            volume_multiplier = 1.50 if USE_XGBOOST else 1.25
    else:
        volume_multiplier = 1.50 if USE_XGBOOST else 1.25

    displacement_multiplier = 0.70 if USE_XGBOOST else 0.50

    df["volume_spike"] = (
        df["volume"]
        >
        df["volume_ma"] * volume_multiplier
    ).astype(int)

    df["body"] = abs(
        df["close"] - df["open"]
    )

    df["displacement"] = (
        df["body"]
        >
        df["atr"] * displacement_multiplier
    ).astype(int)

    return df


# ==========================================================
# MARKET STRUCTURE
# ==========================================================

def bullish_bos(df, structure_lookback=10, confirm_candles=2):
    """
    FIXED: previously swing_high was computed from a window that INCLUDED
    the confirmation candles being tested against it (nearly circular), and
    accepted a bare wick above the level with no close required -- which
    treated liquidity sweeps of equal highs as bullish confirmation instead
    of the trap signal they actually are. Now: swing_high comes from a
    window that excludes the confirmation candles, and both confirmation
    candles must actually CLOSE above the level.
    """
    if len(df) < structure_lookback + confirm_candles + 2:
        return False

    structure_window = df.iloc[-(structure_lookback + confirm_candles):-confirm_candles]
    swing_high = structure_window["high"].max()

    recent_closes = df["close"].iloc[-confirm_candles:]
    return bool((recent_closes > swing_high).all())


def bearish_bos(df, structure_lookback=10, confirm_candles=2):
    """Mirror of bullish_bos -- see docstring there for the bug that was fixed."""
    if len(df) < structure_lookback + confirm_candles + 2:
        return False

    structure_window = df.iloc[-(structure_lookback + confirm_candles):-confirm_candles]
    swing_low = structure_window["low"].min()

    recent_closes = df["close"].iloc[-confirm_candles:]
    return bool((recent_closes < swing_low).all())


def compute_structure_stop(direction, df_entry, entry, atr):
    """Institutional stop: beyond structure + buffer, with a min ATR floor.

    Classic stop-hunt bait was `min(swing*0.9995, entry ± atr*0.7..1.0)` —
    the swing sat *on* equal highs/lows (0.05% "buffer") and the ATR multiples
    lived inside 5m noise. Crypto sweeps those levels by design.

    Policy (always the *wider* room, never a sub-noise ATR clip):
      LONG  SL = min(swing_low - buffer, entry - MIN_SL_ATR * atr)
      SHORT SL = max(swing_high + buffer, entry + MIN_SL_ATR * atr)

    Returns (sl, structure_swing). structure_swing is the raw swing used for
    the level (before buffer) so callers can log / invalidate later.
    """
    atr = float(atr)
    entry = float(entry)
    if atr <= 0 or not np.isfinite(atr) or not np.isfinite(entry):
        raise ValueError(f"invalid atr/entry for stop: atr={atr} entry={entry}")

    lookback = min(STRUCTURE_SWING_LOOKBACK, max(5, len(df_entry) - 2))
    buffer = atr * STRUCTURE_SL_BUFFER_ATR
    floor_dist = atr * MIN_SL_ATR

    # Exclude the forming/signal bar from the swing window (same anti-repaint
    # idea as BOS): stop under "this candle's low" is not structure.
    window = df_entry.iloc[-(lookback + 1):-1]
    if len(window) < 3:
        window = df_entry.iloc[:-1] if len(df_entry) > 1 else df_entry

    if direction == "LONG":
        swing = float(window["low"].min())
        structure_sl = swing - buffer
        atr_floor = entry - floor_dist
        # Lower price = more room for a long stop.
        sl = min(structure_sl, atr_floor)
        # Hard sanity: never above or on entry.
        sl = min(sl, entry - atr * 0.10)
        return float(sl), float(swing)

    if direction == "SHORT":
        swing = float(window["high"].max())
        structure_sl = swing + buffer
        atr_floor = entry + floor_dist
        # Higher price = more room for a short stop.
        sl = max(structure_sl, atr_floor)
        sl = max(sl, entry + atr * 0.10)
        return float(sl), float(swing)

    raise ValueError(f"direction must be LONG or SHORT, got {direction!r}")


def crt_flags(df_5m):
    """Candle Range Theory: has price swept the PREVIOUS completed 1H candle's
    low (or high) and reclaimed back inside its range? The manipulation leg of
    ICT's power-of-three, viewed per HTF candle — sweep below the prior hour's
    low that closes back above it implies expansion toward the other side.

    Purely price-derived (resampled from the 5m frame), so live, backtest and
    pretraining compute it identically. Returns (bull_crt, bear_crt) as 0/1.
    """
    try:
        if len(df_5m) < 15:
            return 0, 0
        h1 = df_5m.resample("1h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        if len(h1) < 2:
            return 0, 0
        # Last resampled bucket is the in-progress hour (built from closed 5m
        # bars); the bucket before it is the previous COMPLETED 1H candle.
        cur, prev = h1.iloc[-1], h1.iloc[-2]
        latest_close = float(df_5m["close"].iloc[-1])
        bull_crt = int(cur["low"] < prev["low"] and latest_close > prev["low"])
        bear_crt = int(cur["high"] > prev["high"] and latest_close < prev["high"])
        return bull_crt, bear_crt
    except Exception:
        return 0, 0


def recent_liquidity_sweep(df_5m, direction, lookback=8, swing_window=12):
    """True if a liquidity sweep + reclaim printed in the last `lookback` bars.

    Desks place limits *after* the sweep is on the chart — they do not wait for
    sweep + displacement + volume + FVG on the *same* candle (that forced late,
    crowded entries). Same-bar sweep still scores; a recent sweep also counts.
    """
    if len(df_5m) < swing_window + lookback + 2:
        return False
    for i in range(1, lookback + 1):
        idx = -i
        start = idx - swing_window
        end = idx
        if abs(start) > len(df_5m):
            continue
        window = df_5m.iloc[start:end]
        if len(window) < 3:
            continue
        candle = df_5m.iloc[idx]
        if direction == "LONG":
            swing_low = window["low"].min()
            if float(candle["low"]) < float(swing_low) and float(candle["close"]) > float(candle["open"]):
                return True
        else:
            swing_high = window["high"].max()
            if float(candle["high"]) > float(swing_high) and float(candle["close"]) < float(candle["open"]):
                return True
    return False


def find_order_block(df_5m, direction, lookback=15):
    """Last opposing candle before a displacement impulse — demand/supply OB mid.

    LONG  → last bearish candle before a bullish impulse (demand).
    SHORT → last bullish candle before a bearish impulse (supply).
    Returns the OB mid price, or None.
    """
    if len(df_5m) < lookback + 3:
        return None
    segment = df_5m.iloc[-(lookback + 1):-1]
    if direction == "LONG":
        for i in range(len(segment) - 1, 0, -1):
            c = segment.iloc[i]
            body = abs(float(c["close"]) - float(c["open"]))
            atr_i = float(c["atr"]) if "atr" in c.index and pd.notna(c["atr"]) else body
            if float(c["close"]) > float(c["open"]) and body > atr_i * 0.55:
                prev = segment.iloc[i - 1]
                if float(prev["close"]) < float(prev["open"]):
                    return (float(prev["open"]) + float(prev["low"])) / 2.0
    else:
        for i in range(len(segment) - 1, 0, -1):
            c = segment.iloc[i]
            body = abs(float(c["close"]) - float(c["open"]))
            atr_i = float(c["atr"]) if "atr" in c.index and pd.notna(c["atr"]) else body
            if float(c["close"]) < float(c["open"]) and body > atr_i * 0.55:
                prev = segment.iloc[i - 1]
                if float(prev["close"]) > float(prev["open"]):
                    return (float(prev["open"]) + float(prev["high"])) / 2.0
    return None


def choose_limit_zone(direction, entry, sl, atr, df_5m, bull_fvg, bear_fvg):
    """Pick the best resting limit: deepest valid of OB / FVG mid / ATR pullback.

    Clamped so risk to SL stays at least 30% of the signal-close risk distance
    (avoids limit-at-SL nonsense) and the order stays on the pullback side of
    current price.
    """
    atr = float(atr)
    entry = float(entry)
    sl = float(sl)
    candidates = []

    if direction == "LONG":
        if bull_fvg and len(df_5m) >= 3:
            gap_bottom = float(df_5m["high"].iloc[-3])
            gap_top = float(df_5m["low"].iloc[-1])
            candidates.append(("fvg", (gap_bottom + gap_top) / 2.0))
        ob = find_order_block(df_5m, "LONG")
        if ob is not None:
            candidates.append(("order_block", float(ob)))
        candidates.append(("atr_pullback", entry - atr * RETRACE_ATR_FRACTION))

        min_limit = sl + 0.30 * (entry - sl)
        valid = [(z, p) for z, p in candidates if min_limit <= p <= entry]
        if not valid:
            limit_price = max(min_limit, min(entry, entry - atr * 0.25))
            zone_type = "atr_pullback"
        else:
            # Deepest valid = best fill for a long (lowest price).
            zone_type, limit_price = min(valid, key=lambda t: t[1])
        limit_price = min(float(limit_price), entry)
        limit_price = max(float(limit_price), min_limit)
    else:
        if bear_fvg and len(df_5m) >= 3:
            gap_top = float(df_5m["low"].iloc[-3])
            gap_bottom = float(df_5m["high"].iloc[-1])
            candidates.append(("fvg", (gap_top + gap_bottom) / 2.0))
        ob = find_order_block(df_5m, "SHORT")
        if ob is not None:
            candidates.append(("order_block", float(ob)))
        candidates.append(("atr_pullback", entry + atr * RETRACE_ATR_FRACTION))

        max_limit = sl - 0.30 * (sl - entry)
        valid = [(z, p) for z, p in candidates if entry <= p <= max_limit]
        if not valid:
            limit_price = min(max_limit, max(entry, entry + atr * 0.25))
            zone_type = "atr_pullback"
        else:
            zone_type, limit_price = max(valid, key=lambda t: t[1])
        limit_price = max(float(limit_price), entry)
        limit_price = min(float(limit_price), max_limit)

    return float(limit_price), zone_type


# ==========================================================
# MAIN SIGNAL ENGINE
# ==========================================================

def get_signal(symbol, df_15m, df_5m):

    try:

        if len(df_15m) < 30:

            return None

        if len(df_5m) < 30:

            return None

        df_15m = calculate_features(df_15m)
        df_5m = calculate_features(df_5m)

        trend_bull = bullish_bos(df_15m)
        trend_bear = bearish_bos(df_15m)

        latest = df_5m.iloc[-1]

        atr = latest["atr"]

        if pd.isna(atr):

            return None

        if atr <= 0:

            return None

        swing_low = df_5m["low"].iloc[-10:-1].min()
        swing_high = df_5m["high"].iloc[-10:-1].max()

        # ------------------------------------
        # Liquidity Sweep (same-bar OR recent window)
        # ------------------------------------
        # Same-bar sweep still scores; a recent sweep in the last ~8 bars also
        # counts so we can rest a limit without requiring every confluence on
        # this exact candle.

        if USE_XGBOOST:

            bull_sweep_now = (
                latest["low"] < swing_low
                and latest["close"] > latest["open"]
            )

            bear_sweep_now = (
                latest["high"] > swing_high
                and latest["close"] < latest["open"]
            )

        else:

            bull_sweep_now = (
                latest["low"] <= swing_low * 1.0002
                and latest["close"] > latest["open"]
            )

            bear_sweep_now = (
                latest["high"] >= swing_high * 0.9998
                and latest["close"] < latest["open"]
            )

        bull_sweep_recent = recent_liquidity_sweep(df_5m, "LONG")
        bear_sweep_recent = recent_liquidity_sweep(df_5m, "SHORT")
        bull_sweep = bull_sweep_now or bull_sweep_recent
        bear_sweep = bear_sweep_now or bear_sweep_recent

        # ------------------------------------
        # Fair Value Gap (current or still-open prior gap)
        # ------------------------------------

        bull_fvg = (
            df_5m["low"].iloc[-1]
            > df_5m["high"].iloc[-3]
        )
        if not bull_fvg and len(df_5m) >= 5:
            for k in range(2, 5):
                if df_5m["low"].iloc[-k] > df_5m["high"].iloc[-(k + 2)]:
                    if float(latest["close"]) > float(df_5m["high"].iloc[-(k + 2)]):
                        bull_fvg = True
                        break

        bear_fvg = (
            df_5m["high"].iloc[-1]
            < df_5m["low"].iloc[-3]
        )
        if not bear_fvg and len(df_5m) >= 5:
            for k in range(2, 5):
                if df_5m["high"].iloc[-k] < df_5m["low"].iloc[-(k + 2)]:
                    if float(latest["close"]) < float(df_5m["low"].iloc[-(k + 2)]):
                        bear_fvg = True
                        break

        # Candle Range Theory: now a soft setup edge (was feature-only).
        bull_crt, bear_crt = crt_flags(df_5m)

        # Soft institutional score: HTF bias + edges. Volume/displacement are
        # bonuses, NOT hard requirements — rest the limit when structure is
        # there and wait; do not need a volume spike on the order bar.
        score = 0

        if trend_bull or trend_bear:
            score += 20

        if bull_sweep or bear_sweep:
            score += 20

        if bull_fvg or bear_fvg:
            score += 15

        if bull_crt or bear_crt:
            score += 15

        if latest["volume_spike"]:
            score += 15

        if latest["displacement"]:
            score += 15

        _ob_long = find_order_block(df_5m, "LONG")
        _ob_short = find_order_block(df_5m, "SHORT")
        if (trend_bull and _ob_long is not None) or (trend_bear and _ob_short is not None):
            score += 10

        entry = float(latest["close"])

        # UTC, not local time. Candle timestamps and the session buckets
        # (London/NY/Asian) are all defined in UTC; datetime.now() (naive
        # local) would mislabel every session feature on any non-UTC host.
        # NOW_FN is overridable so the backtester can replay historical bars.
        now = NOW_FN()

        hour = now.hour

        day_of_week = now.weekday()

        history = get_trade_history()

        # ==========================================================
        # Historical Context
        # ==========================================================

        context = (
            calculate_historical_context(history)
            if len(history) >= 5
            else {
                "recent_win_rate": 0.50,
                "streak_count": 0,
                "cumulative_pnl": 0,
                "current_dd_pct": 0,
            }
        )

        # ==========================================================
        # Contextual raw features (regime / volatility / distances)
        # ----------------------------------------------------------
        # Computed once here and stored on the returned signal so that
        # TRAINING (extract_pro_features_from_trade reading the persisted
        # trade) and INFERENCE (below) derive identical feature vectors.
        # Previously none of these were persisted, so the trainer read the
        # defaults every time and these features were dead constants.
        # ==========================================================

        market_regime = detect_market_regime(df_5m)
        atr_percentile = calculate_atr_percentile(df_5m["atr"].dropna(), atr)

        def _rel(a, b):
            b = float(b) if pd.notna(b) else entry
            return float((a - b) / b) if b else 0.0

        distance_to_ema20 = _rel(entry, latest.get("ema20", entry))
        distance_to_ema50 = _rel(entry, latest.get("ema50", entry))
        distance_to_vwap = _rel(entry, latest.get("vwap", entry))
        distance_to_prev_high = _rel(swing_high, entry)
        distance_to_prev_low = _rel(swing_low, entry)

        # ==========================================================
        # Market context (funding / OI / BTC trend / spread) + per-symbol form
        # ----------------------------------------------------------
        # Only fetched once a real setup exists (trend + minimum confluence) so
        # a 30-symbol scan doesn't triple its API calls on non-setups. Values
        # are persisted on the signal so training sees exactly what inference
        # saw. All None-safe: an API hiccup degrades to neutral, never blocks.
        # ==========================================================

        market_ctx = {}
        if (trend_bull or trend_bear) and score >= 30:
            try:
                market_ctx = MARKET_CONTEXT_FN(symbol) or {}
            except Exception as e:
                logger.debug(f"market context fetch failed for {symbol}: {e}")

        # Per-symbol win rate over its last 20 closed trades (0.5 = neutral /
        # not enough data). Lets the model learn "this strategy works on SOL
        # but not on THIS thin alt" instead of treating every pair identically.
        symbol_closed = [
            t for t in history
            if t.get("symbol") == symbol and t.get("status") in ("WIN", "LOSS")
        ][-20:]
        symbol_win_rate = (
            sum(1 for t in symbol_closed if t.get("status") == "WIN") / len(symbol_closed)
            if len(symbol_closed) >= 3
            else 0.5
        )

        # ==========================================================
        # Dynamic Stop Loss / Take Profit
        # ----------------------------------------------------------
        # RR target still adapts to vol/score. Stop placement is
        # structure-first (compute_structure_stop) — never the old
        # "tight ATR multiple sitting on the swing" stop-hunt bait.
        # ==========================================================

        atr_average = df_5m["atr"].dropna().tail(30).mean()

        if pd.isna(atr_average):
            atr_average = atr

        if USE_XGBOOST:

            if atr > atr_average * 1.30:
                rr = 2.50

            elif atr < atr_average * 0.70:
                rr = 1.80

            else:
                rr = 2.00

        else:

            # Dynamic RR based on soft setup quality
            if score >= 70:
                rr = 2.0
            elif score >= 45:
                rr = 1.75
            else:
                rr = 1.5

        # Soft gate early: no HTF bias → no trade. Limit mode only needs a
        # minimal edge score, not full confluence alignment.
        if not trend_bull and not trend_bear:
            return None

        if ENTRY_MODE == "limit" and score < LIMIT_MIN_SETUP_SCORE:
            return None

        direction = "LONG" if trend_bull else "SHORT"

        # Structure stop from signal close first (risk geometry).
        sl, structure_swing = compute_structure_stop(direction, df_5m, entry, atr)

        if direction == "LONG":
            tp = entry + ((entry - sl) * rr)
        else:
            tp = entry - ((sl - entry) * rr)

        # ==========================================================
        # Prediction limit zone (OB / FVG / ATR pullback)
        # ----------------------------------------------------------
        # Desk behaviour: identify the zone, rest a limit, wait. Not chase
        # the displacement close. main.py uses limit_price when ENTRY_MODE
        # is "limit"; market mode keeps the signal close as entry.
        # ==========================================================

        limit_price, zone_type = choose_limit_zone(
            direction, entry, sl, atr, df_5m, bull_fvg, bear_fvg
        )

        # Recompute SL/TP at the *limit* so risk is measured from where we
        # actually intend to get filled (not the chase price).
        if ENTRY_MODE == "limit":
            sl, structure_swing = compute_structure_stop(
                direction, df_5m, limit_price, atr
            )
            if direction == "LONG":
                sl = min(sl, limit_price - atr * MIN_SL_ATR * 0.85)
                tp = limit_price + ((limit_price - sl) * rr)
            else:
                sl = max(sl, limit_price + atr * MIN_SL_ATR * 0.85)
                tp = limit_price - ((sl - limit_price) * rr)
            invalidation_price = float(structure_swing)
            trade_entry = float(limit_price)
        else:
            invalidation_price = float(structure_swing)
            trade_entry = float(entry)

        # ==========================================================
        # Risk Metrics (from the economics we will actually trade)
        # ==========================================================

        risk_pct = abs(trade_entry - sl) / max(trade_entry, 0.0001)

        reward_pct = abs(tp - trade_entry) / max(trade_entry, 0.0001)

        adversity_ratio = (
            risk_pct /
            max(reward_pct, 0.0001)
        )

        risk_reward = (
            abs(tp - trade_entry)
            /
            max(abs(trade_entry - sl), 0.0001)
        )

        # ==========================================================
        # Signal snapshot — the single canonical record of everything this
        # setup looked like at entry. It is (a) fed to the SAME featurizer the
        # trainer uses, and (b) persisted verbatim on the returned signal, so
        # training and inference can never drift apart.
        # ==========================================================

        # Diagnostic fields (zone_type / invalidation / structure_swing /
        # setup_score) are ignored by the featurizer — keeps train/serve
        # feature contract stable. Do NOT add them as model features here.
        signal_snapshot = {
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "signal_close": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "structure_swing": float(structure_swing),
            "invalidation_price": float(invalidation_price),
            "zone_type": zone_type,
            "setup_score": int(score),
            "volume_spike": int(latest["volume_spike"]),
            "displacement": int(latest["displacement"]),
            "sweep": int(bull_sweep if trend_bull else bear_sweep),
            "fvg": int(bull_fvg if trend_bull else bear_fvg),
            "crt": int(bull_crt if trend_bull else bear_crt),
            "atr": float(atr),
            "body": float(latest["body"]) if pd.notna(latest["body"]) else 0.0,
            "volume": float(latest["volume"]) if pd.notna(latest["volume"]) else 0.0,
            "volume_ma": float(latest["volume_ma"]) if pd.notna(latest["volume_ma"]) else 0.0,
            "hour": hour,
            "day_of_week": day_of_week,
            "market_regime": market_regime,
            "atr_percentile": float(atr_percentile),
            "distance_to_ema20": distance_to_ema20,
            "distance_to_ema50": distance_to_ema50,
            "distance_to_vwap": distance_to_vwap,
            "distance_to_prev_high": distance_to_prev_high,
            "distance_to_prev_low": distance_to_prev_low,
            "risk_reward": risk_reward,
            "rr_multiplier": rr,
            "limit_price": float(limit_price),
            # Market context — persisted so training sees what inference saw.
            "funding_rate": market_ctx.get("funding_rate"),
            "oi_change_pct": market_ctx.get("oi_change_pct"),
            "btc_trend": market_ctx.get("btc_trend"),
            "spread_pct": market_ctx.get("spread_pct"),
            "fng": market_ctx.get("fng"),
            "symbol_win_rate": float(symbol_win_rate),
        }

        # ==========================================================
        # AI Probability
        # ----------------------------------------------------------
        # Uses extract_pro_features_from_trade — the EXACT same function the
        # trainer uses to build its training matrix. Previously inference
        # hand-built a smaller, differently-named dict and get_xgboost_
        # probability filled every unrecognized feature with 0, so the model
        # was scored on a feature vector it never saw in training (train/serve
        # skew). Same function in both paths guarantees parity by construction.
        # ==========================================================

        ai_prob = 50.0
        expected_r = None
        ai_weight = 0.0

        if USE_XGBOOST:
            features = extract_pro_features_from_trade(
                signal_snapshot, context, regime=market_regime
            )
            ai_prob = get_xgboost_probability(features)
            # Auxiliary regression: how many R the model expects this setup to
            # realize. Used as an entry filter below. None until the model has
            # enough data to train.
            expected_r = get_expected_r(features)
            signal_snapshot["expected_r"] = expected_r

        # ==========================================================
        # Confidence Engine
        # ==========================================================

        n_real_closed = len([
            t for t in history
            if t.get("status") in ("WIN", "LOSS")
            and t.get("source", "live") != "backtest"
        ])

        if USE_XGBOOST:

            # AI Mode: blend soft SMC score with the model probability.
            # Limit mode uses a lower dynamic floor so predictions can rest;
            # expected-R (when live sample is large enough) still filters.
            #
            # The model's vote is scaled by how many REAL closed trades it has
            # learned from (ai_blend_weight): a 42-trade model gets ~4% say,
            # not the full 40% — trust is earned with sample size, not granted
            # the moment a model file exists.
            ai_weight = ai_blend_weight(n_real_closed)

            final_confidence = int(
                (score * (1 - ai_weight)) +
                (ai_prob * ai_weight)
            )

            confidence_required = get_dynamic_confidence_threshold(
                regime=market_regime,
                atr_percentile=atr_percentile,
                recent_win_rate=context.get("recent_win_rate", 0.5),
                entry_mode=ENTRY_MODE,
            )

        else:

            # Pure SMC: limit mode places predictions with a soft bar;
            # market mode keeps the stricter CONFIDENCE_REQUIRED_SMC.
            final_confidence = score
            confidence_required = (
                CONFIDENCE_REQUIRED_LIMIT
                if ENTRY_MODE == "limit"
                else CONFIDENCE_REQUIRED_SMC
            )

        # Confidence bonus: reward stacked edges (not required)
        if (bull_sweep and bull_fvg) or (bear_sweep and bear_fvg):
            final_confidence += 5

        if latest["volume_spike"] and latest["displacement"]:
            final_confidence += 5

        if (bull_crt and bull_sweep) or (bear_crt and bear_sweep):
            final_confidence += 4

        if zone_type in ("order_block", "fvg"):
            final_confidence += 3

        final_confidence = max(
            0,
            min(final_confidence, 100)
        )

        # ==========================================================
        # Logging — ONE compact line per evaluated setup. The old 25-line
        # banner multiplied by every trending symbol per scan (and by every
        # bar in a backfill) blew straight through Railway's 500 logs/sec
        # rate limit and got messages dropped.
        # Reads: symbol, direction, confluences hit, confidence vs required.
        # ==========================================================

        confluences = "".join([
            "V" if latest["volume_spike"] else "-",
            "D" if latest["displacement"] else "-",
            "S" if (bull_sweep or bear_sweep) else "-",
            "F" if (bull_fvg or bear_fvg) else "-",
            "C" if (bull_crt or bear_crt) else "-",
        ])
        passed = final_confidence >= confidence_required
        logger.info(
            f"{'🎯' if passed else '·'} {symbol} {direction} "
            f"| smc {score} [{confluences}] zone={zone_type} "
            f"| ai {ai_prob:.0f}%@{ai_weight:.0%} "
            f"| conf {final_confidence}/{confidence_required} "
            f"| rr {risk_reward:.1f} | {market_regime}"
        )

        # ==========================================================
        # Signal Hash
        # ==========================================================

        candle_time = (
            df_5m.index[-1]
            if hasattr(df_5m.index, "__len__")
            else datetime.now(timezone.utc)
        )

        # Symbol is part of the hash now. Without it, two different pairs that
        # print the same direction on the same candle timestamp collide, and
        # the second one is silently dropped as a "duplicate" — suppressing
        # real signals across the whole scan set.
        signal_hash = f"{symbol}_{candle_time}_{direction}"

        # Cooldown: prevent duplicate signals on the same candle for this pair.
        if signal_hash in recent_signals:
            return None

        # ==========================================================
        # SIGNAL — direction already encoded in signal_snapshot, so a single
        # return covers both LONG and SHORT (exactly one of trend_bull /
        # trend_bear is True to reach here; the SL/TP block returns None
        # otherwise).
        # ==========================================================

        # Expected-R filter (AI mode only): once the regression model exists,
        # reject setups it expects to realize less than MIN_EXPECTED_R, even if
        # the win probability clears the confidence bar — a low win rate on
        # good R can be fine, but a decent win rate on poor R is not. Skipped
        # (expected_r is None) until the model has trained.
        #
        # Also gated on >=60 REAL closed trades: a model warmed up mostly on
        # backtest rows must not VETO live entries — that would bias the very
        # training data being collected (setups it wrongly dislikes would
        # never get a real outcome to correct it with).
        if (
            USE_XGBOOST
            and expected_r is not None
            and n_real_closed >= 60
            and expected_r < MIN_EXPECTED_R
        ):
            logger.info(
                f"↩️ {symbol}: expected R {expected_r:.2f} < {MIN_EXPECTED_R} — skipping"
            )
            return None

        if final_confidence >= confidence_required:

            recent_signals.add(signal_hash)
            return {
                **signal_snapshot,
                "confidence": final_confidence,
                "ai_prob": ai_prob,
                "signal_hash": signal_hash,
                "recent_win_rate": context.get("recent_win_rate", 0.5),
                "streak_count": context.get("streak_count", 0),
            }

        return None

    except Exception as e:

        logger.exception(
            f"Signal error: {e}"
        )

        return None
