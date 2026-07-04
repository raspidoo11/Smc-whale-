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
from config import MIN_EXPECTED_R

logger = logging.getLogger(__name__)

USE_XGBOOST = os.getenv("USE_XGBOOST", "false").lower() == "true"

# Time source seam. Live trading uses wall-clock UTC; the backtester overrides
# this to the candle's timestamp so session/hour features reflect the bar being
# replayed, not "now". Keeps a single get_signal() for both paths.
NOW_FN = lambda: datetime.now(timezone.utc)

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
        # Liquidity Sweep
        # ------------------------------------

        if USE_XGBOOST:

            bull_sweep = (
                latest["low"] < swing_low
                and latest["close"] > latest["open"]
            )

            bear_sweep = (
                latest["high"] > swing_high
                and latest["close"] < latest["open"]
            )

        else:

            bull_sweep = (
                latest["low"] <= swing_low * 1.0002
                and latest["close"] > latest["open"]
            )

            bear_sweep = (
                latest["high"] >= swing_high * 0.9998
                and latest["close"] < latest["open"]
            )

        # ------------------------------------
        # Fair Value Gap
        # ------------------------------------

        bull_fvg = (
            df_5m["low"].iloc[-1]
            > df_5m["high"].iloc[-3]
        )

        bear_fvg = (
            df_5m["high"].iloc[-1]
            < df_5m["low"].iloc[-3]
        )

        score = 0

        if latest["volume_spike"]:

            score += 25

        if latest["displacement"]:

            score += 25

        if trend_bull:

            score += 20

        if trend_bear:

            score += 20

        if bull_sweep or bear_sweep:

            score += 20

        if bull_fvg or bear_fvg:

            score += 15

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
        # Dynamic Stop Loss / Take Profit
        # ==========================================================

        atr_average = df_5m["atr"].dropna().tail(30).mean()

        if pd.isna(atr_average):
            atr_average = atr

        if USE_XGBOOST:

            if atr > atr_average * 1.30:
                sl_multiplier = 1.00
                rr = 2.50

            elif atr < atr_average * 0.70:
                sl_multiplier = 0.70
                rr = 1.80

            else:
                sl_multiplier = 0.85
                rr = 2.00

        else:

            # Easier targets while collecting AI data
            # Dynamic RR based on SMC score (higher confluence = better RR)
            sl_multiplier = 0.80
            if score >= 80:
                rr = 2.0
            elif score >= 60:
                rr = 1.75
            else:
                rr = 1.5

        # ==========================================================
        # LONG
        # ==========================================================

        if trend_bull:

            swing_low = df_5m["low"].iloc[-8:-1].min()

            sl = min(
                swing_low * 0.9995,
                entry - atr * sl_multiplier
            )

            tp = entry + ((entry - sl) * rr)

        # ==========================================================
        # SHORT
        # ==========================================================

        elif trend_bear:

            swing_high = df_5m["high"].iloc[-8:-1].max()

            sl = max(
                swing_high * 1.0005,
                entry + atr * sl_multiplier
            )

            tp = entry - ((sl - entry) * rr)

        else:

            return None

        # ==========================================================
        # Risk Metrics
        # ==========================================================

        risk_pct = abs(entry - sl) / max(entry, 0.0001)

        reward_pct = abs(tp - entry) / max(entry, 0.0001)

        adversity_ratio = (
            risk_pct /
            max(reward_pct, 0.0001)
        )

        risk_reward = (
            abs(tp - entry)
            /
            max(abs(entry - sl), 0.0001)
        )

        # ==========================================================
        # Signal snapshot — the single canonical record of everything this
        # setup looked like at entry. It is (a) fed to the SAME featurizer the
        # trainer uses, and (b) persisted verbatim on the returned signal, so
        # training and inference can never drift apart.
        # ==========================================================

        direction = "LONG" if trend_bull else "SHORT"

        signal_snapshot = {
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "sl": float(sl),
            "tp": float(tp),
            "volume_spike": int(latest["volume_spike"]),
            "displacement": int(latest["displacement"]),
            "sweep": int(bull_sweep if trend_bull else bear_sweep),
            "fvg": int(bull_fvg if trend_bull else bear_fvg),
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

        if USE_XGBOOST:

            # AI Mode: blend SMC score with the model probability, and require a
            # confidence that adapts to market conditions (looser in clean
            # trends, stricter in chop/high-vol and after a losing run) instead
            # of a hardcoded constant.
            final_confidence = int(
                (score * 0.60) +
                (ai_prob * 0.40)
            )

            confidence_required = get_dynamic_confidence_threshold(
                regime=market_regime,
                atr_percentile=atr_percentile,
                recent_win_rate=context.get("recent_win_rate", 0.5),
            )

        else:

            # Pure SMC Mode
            # Generate more trades for AI training
            final_confidence = score

            confidence_required = 40

        # Confidence Bonus: reward setups with multiple confluences
        if (bull_sweep and bull_fvg) or (bear_sweep and bear_fvg):
            final_confidence += 5

        if latest["volume_spike"] and latest["displacement"]:
            final_confidence += 5

        final_confidence = max(
            0,
            min(final_confidence, 100)
        )

        # ==========================================================
        # Detailed Logging
        # ==========================================================

        logger.info(

            f"""
================ SIGNAL SCAN ================

Symbol            : {symbol}

Mode              : {'AI' if USE_XGBOOST else 'SMC'}

Trend Bull        : {trend_bull}
Trend Bear        : {trend_bear}

Volume Spike      : {latest['volume_spike']}
Displacement      : {latest['displacement']}
Liquidity Sweep   : {bull_sweep or bear_sweep}
Fair Value Gap    : {bull_fvg or bear_fvg}

SMC Score         : {score}
AI Probability    : {ai_prob:.2f}

Risk Reward       : {risk_reward:.2f}

Final Confidence  : {final_confidence}
Required          : {confidence_required}

=============================================
"""
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
        if USE_XGBOOST and expected_r is not None and expected_r < MIN_EXPECTED_R:
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
