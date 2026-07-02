import pandas as pd
import logging
import os
from datetime import datetime
from trade_manager import get_trade_history
from xgboost_trainer import (
    calculate_historical_context,
    get_xgboost_probability,
)

logger = logging.getLogger(__name__)

USE_XGBOOST = os.getenv("USE_XGBOOST", "false").lower() == "true"

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

def get_signal(df_15m, df_5m):

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

        now = datetime.now()

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

        ai_prob = 50.0

        # ==========================================================
        # XGBoost Prediction
        # ==========================================================

        if USE_XGBOOST:

            trade_features = {

                "volume_spike": latest["volume_spike"],

                "displacement": latest["displacement"],

                "trend_bull": int(trend_bull),

                "sweep": int(
                    bull_sweep or bear_sweep
                ),

                "fvg": int(
                    bull_fvg or bear_fvg
                ),

                "atr": float(atr),

                "risk_reward": risk_reward,

                "hour": hour,

                "day_of_week": day_of_week,

                "body_ratio":
                    latest["body"] /
                    max(atr, 0.0001),

                "volume_strength":
                    latest["volume"] /
                    max(latest["volume_ma"], 0.0001),

                "atr_expansion": float(atr),

                "confluence_count": sum([
                    latest["volume_spike"],
                    latest["displacement"],
                    bull_sweep or bear_sweep,
                    bull_fvg or bear_fvg
                ]),

                "is_london_open":
                    1 if 7 <= hour <= 11 else 0,

                "is_ny_open":
                    1 if 12 <= hour <= 16 else 0,

                "is_asian":
                    1 if (hour >= 22 or hour <= 6) else 0,

                "is_overlap":
                    1 if 12 <= hour <= 15 else 0,

                "is_quiet_time":
                    1 if 17 <= hour <= 21 else 0,

                "is_monday":
                    1 if day_of_week == 0 else 0,

                "is_friday":
                    1 if day_of_week == 4 else 0,

                "risk_pct": risk_pct,

                "reward_pct": reward_pct,

                "adversity_ratio": adversity_ratio,

                "recent_win_rate":
                    context.get(
                        "recent_win_rate",
                        0.5
                    ),

                "streak_count":
                    context.get(
                        "streak_count",
                        0
                    ),

                "is_hot_streak":
                    1 if context.get(
                        "streak_count",
                        0
                    ) > 0 else 0,

                "cumulative_pnl":
                    context.get(
                        "cumulative_pnl",
                        0
                    ),

                "current_dd_pct":
                    context.get(
                        "current_dd_pct",
                        0
                    ),

                "volume_x_displacement":

                    latest["volume_spike"]
                    *
                    latest["displacement"],

                "sweep_x_fvg":

                    int(bull_sweep or bear_sweep)
                    *
                    int(bull_fvg or bear_fvg),

                "volatility_x_risk":

                    float(atr)
                    *
                    risk_pct,

            }

            ai_prob = get_xgboost_probability(
                trade_features
            )

        # ==========================================================
        # Confidence Engine
        # ==========================================================

        if USE_XGBOOST:

            # AI Mode
            # Let the model have slightly more influence
            final_confidence = int(
                (score * 0.45) +
                (ai_prob * 0.55)
            )

            confidence_required = 55

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
            else datetime.now()
        )

        signal_hash = (
            f"{candle_time}_"
            f"{'LONG' if trend_bull else 'SHORT'}"
        )

        # Cooldown: prevent duplicate signals on the same candle (for this pair/context)
        if signal_hash in recent_signals:
            return None

        # ==========================================================
        # LONG SIGNAL
        # ==========================================================

        if trend_bull and final_confidence >= confidence_required:

            recent_signals.add(signal_hash)
            return {

                "direction": "LONG",

                "confidence": final_confidence,

                "entry": entry,

                "sl": float(sl),

                "tp": float(tp),

                "ai_prob": ai_prob,

                "signal_hash": signal_hash,

                "volume_spike": int(latest["volume_spike"]),

                "displacement": int(latest["displacement"]),

                "sweep": int(bull_sweep),

                "fvg": int(bull_fvg),

                "atr": float(atr),

                "risk_reward": risk_reward,

                "hour": hour,

                "day_of_week": day_of_week,

                "recent_win_rate": context.get(
                    "recent_win_rate",
                    0.5
                ),

                "streak_count": context.get(
                    "streak_count",
                    0
                ),

            }

        # ==========================================================
        # SHORT SIGNAL
        # ==========================================================

        if trend_bear and final_confidence >= confidence_required:

            recent_signals.add(signal_hash)
            return {

                "direction": "SHORT",

                "confidence": final_confidence,

                "entry": entry,

                "sl": float(sl),

                "tp": float(tp),

                "ai_prob": ai_prob,

                "signal_hash": signal_hash,

                "volume_spike": int(latest["volume_spike"]),

                "displacement": int(latest["displacement"]),

                "sweep": int(bear_sweep),

                "fvg": int(bear_fvg),

                "atr": float(atr),

                "risk_reward": risk_reward,

                "hour": hour,

                "day_of_week": day_of_week,

                "recent_win_rate": context.get(
                    "recent_win_rate",
                    0.5
                ),

                "streak_count": context.get(
                    "streak_count",
                    0
                ),

            }

        return None

    except Exception as e:

        logger.exception(
            f"Signal error: {e}"
        )

        return None
