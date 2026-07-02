import pandas as pd
import logging
import os
from datetime import datetime, timezone
from news_filter import is_high_impact_news_time
from trade_manager import get_trade_history
from xgboost_trainer import (
    calculate_historical_context,
    get_xgboost_probability,
    detect_market_regime,
    get_dynamic_confidence_threshold,
    get_ai_risk_percent
)

logger = logging.getLogger(__name__)

USE_XGBOOST = os.getenv("USE_XGBOOST", "false").lower() == "true"

# ==================== SESSION CONFIG ====================
TRADE_ASIAN = os.getenv("TRADE_ASIAN", "true").lower() == "true"
TRADE_LONDON = os.getenv("TRADE_LONDON", "true").lower() == "true"
TRADE_NEWYORK = os.getenv("TRADE_NEWYORK", "true").lower() == "true"

OVERLAP_BONUS = int(os.getenv("OVERLAP_BONUS", 20))
LONDON_BONUS = int(os.getenv("LONDON_BONUS", 10))
NEWYORK_BONUS = int(os.getenv("NEWYORK_BONUS", 12))
ASIAN_PENALTY = int(os.getenv("ASIAN_PENALTY", -10))


def get_session_bonus():
    now = datetime.now(timezone.utc)
    hour = now.hour

    is_london = 7 <= hour < 12
    is_newyork = 12 <= hour < 17
    is_overlap = 12 <= hour < 15
    is_asian = (hour >= 22 or hour < 7)

    bonus = 0
    if is_overlap:
        bonus = OVERLAP_BONUS
    elif is_newyork:
        bonus = NEWYORK_BONUS
    elif is_london:
        bonus = LONDON_BONUS
    elif is_asian:
        bonus = ASIAN_PENALTY

    return bonus, {
        "is_london": is_london,
        "is_newyork": is_newyork,
        "is_overlap": is_overlap,
        "is_asian": is_asian
    }


def calculate_features(df):
    df = df.copy()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["volume_spike"] = (df["volume"] > df["volume_ma"] * 1.5).astype(int)
    df["body"] = abs(df["close"] - df["open"])
    df["displacement"] = (df["body"] > df["atr"] * 0.7).astype(int)
    return df


def bullish_bos(df):
    return df["close"].iloc[-1] > df["high"].iloc[-10:-1].max()


def bearish_bos(df):
    return df["close"].iloc[-1] < df["low"].iloc[-10:-1].min()


def get_signal(df_15m, df_5m):
    try:
        if len(df_15m) < 30 or len(df_5m) < 30:
            return None

        df_15m = calculate_features(df_15m)
        df_5m = calculate_features(df_5m)

        trend_bull = bullish_bos(df_15m)
        trend_bear = bearish_bos(df_15m)

        latest = df_5m.iloc[-1]
        atr = latest["atr"]

        if pd.isna(atr) or atr <= 0:
            return None
        # ==================== NEWS FILTER ====================
        if is_high_impact_news_time(minutes_before=30, minutes_after=30):
            logger.info("🚫 Skipping signal - High impact news window active")
            return None

        bull_sweep = (latest["low"] < df_5m["low"].iloc[-10:-1].min() and latest["close"] > latest["open"])
        bear_sweep = (latest["high"] > df_5m["high"].iloc[-10:-1].max() and latest["close"] < latest["open"])

        bull_fvg = df_5m["low"].iloc[-1] > df_5m["high"].iloc[-3]
        bear_fvg = df_5m["high"].iloc[-1] < df_5m["low"].iloc[-3]

        score = 0
        if latest["volume_spike"] == 1: score += 25
        if latest["displacement"] == 1: score += 25
        if trend_bull: score += 20
        if trend_bear: score += 20
        if bull_sweep or bear_sweep: score += 20
        if bull_fvg or bear_fvg: score += 15

        entry = float(latest["close"])

        # Dynamic SL/TP
        atr_series = df_5m["atr"].dropna()
        atr_avg = atr_series.tail(30).mean() if len(atr_series) >= 10 else atr

        if atr > atr_avg * 1.3:
            sl_multiplier, rr_multiplier = 1.0, 2.8
        elif atr < atr_avg * 0.7:
            sl_multiplier, rr_multiplier = 0.65, 1.6
        else:
            sl_multiplier, rr_multiplier = 0.85, 2.0

        if trend_bull:
            swing_low = df_5m["low"].iloc[-8:-1].min()
            sl = min(swing_low * 0.9995, entry - atr * sl_multiplier)
            tp = entry + (entry - sl) * rr_multiplier
        elif trend_bear:
            swing_high = df_5m["high"].iloc[-8:-1].max()
            sl = max(swing_high * 1.0005, entry + atr * sl_multiplier)
            tp = entry - (sl - entry) * rr_multiplier
        else:
            return None

        # Session handling
        session_bonus, session_info = get_session_bonus()

        if (session_info["is_asian"] and not TRADE_ASIAN) or \
           (session_info["is_london"] and not TRADE_LONDON) or \
           (session_info["is_newyork"] and not TRADE_NEWYORK):
            return None

        risk_reward = abs(tp - entry) / max(abs(entry - sl), 0.0001)

        now = datetime.now(timezone.utc)
        hour = now.hour
        day_of_week = now.weekday()

        history = get_trade_history()
        context = calculate_historical_context(history) if len(history) >= 5 else {
            "recent_win_rate": 0.5, "streak_count": 0, "cumulative_pnl": 0, "current_dd_pct": 0
        }

        # Detect regime for AI
        regime = detect_market_regime(df_5m)
        atr_percentile = calculate_atr_percentile(df_5m["atr"].dropna(), atr)

        ai_prob = 50.0
        if USE_XGBOOST:
            trade_features = {
                "volume_spike": latest["volume_spike"],
                "displacement": latest["displacement"],
                "trend_bull": 1 if trend_bull else 0,
                "sweep": 1 if (bull_sweep or bear_sweep) else 0,
                "fvg": 1 if (bull_fvg or bear_fvg) else 0,
                "atr": float(atr),
                "risk_reward": risk_reward,
                "hour": hour,
                "day_of_week": day_of_week,
                "body_ratio": latest["body"] / max(atr, 0.0001),
                "volume_strength": latest["volume"] / max(latest["volume_ma"], 0.0001),
                "atr_expansion": float(atr),
                "confluence_count": sum([latest["volume_spike"], latest["displacement"], bull_sweep or bear_sweep, bull_fvg or bear_fvg]),
                "is_london_open": 1 if 7 <= hour <= 11 else 0,
                "is_ny_open": 1 if 12 <= hour <= 16 else 0,
                "is_asian": 1 if (hour >= 22 or hour <= 6) else 0,
                "is_overlap": 1 if 8 <= hour <= 11 else 0,
                "is_quiet_time": 1 if 17 <= hour <= 21 else 0,
                "is_monday": 1 if day_of_week == 0 else 0,
                "is_friday": 1 if day_of_week == 4 else 0,
                "risk_pct": abs(entry - sl) / entry,
                "reward_pct": abs(tp - entry) / entry,
                "adversity_ratio": abs(entry - sl) / max(abs(tp - entry), 0.0001),
                "recent_win_rate": context.get("recent_win_rate", 0.5),
                "streak_count": context.get("streak_count", 0),
                "is_hot_streak": 1 if context.get("streak_count", 0) > 0 else 0,
                "cumulative_pnl": context.get("cumulative_pnl", 0),
                "current_dd_pct": context.get("current_dd_pct", 0),
                "volume_x_displacement": latest["volume_spike"] * latest["displacement"],
                "sweep_x_fvg": (1 if (bull_sweep or bear_sweep) else 0) * (1 if (bull_fvg or bear_fvg) else 0),
                "volatility_x_risk": float(atr) * (abs(entry - sl) / entry),
                "atr_percentile": atr_percentile,
                "is_high_volatility": 1 if atr_percentile > 70 else 0,
                "is_low_volatility": 1 if atr_percentile < 30 else 0,
                "regime_trending": 1 if regime == "trending" else 0,
                "regime_ranging": 1 if regime == "ranging" else 0,
                "regime_volatile": 1 if regime == "volatile" else 0,
            }

            ai_prob = get_xgboost_probability(trade_features, recent_win_rate=context.get("recent_win_rate", 0.5))

        # Dynamic confidence threshold
        min_confidence = get_dynamic_confidence_threshold(
            regime=regime,
            atr_percentile=atr_percentile,
            recent_win_rate=context.get("recent_win_rate", 0.5)
        )

        final_confidence = int((score * 0.5) + (ai_prob * 0.5))
        final_confidence += session_bonus
        final_confidence = max(0, min(final_confidence, 100))

        # Generate signal_hash
        candle_timestamp = df_5m.index[-1].timestamp() if hasattr(df_5m.index[-1], 'timestamp') else int(now.timestamp())
        signal_hash = f"{latest.name}_{'LONG' if trend_bull else 'SHORT'}_{int(candle_timestamp)}"

        if (trend_bull or trend_bear) and final_confidence >= min_confidence:
            return {
                "direction": "LONG" if trend_bull else "SHORT",
                "confidence": final_confidence,
                "entry": entry,
                "sl": float(sl),
                "tp": float(tp),
                "ai_prob": ai_prob,
                "signal_hash": signal_hash,
                "session_bonus": session_bonus,
                "market_regime": regime,
                "atr_percentile": atr_percentile,
                **session_info
            }

        return None

    except Exception as e:
        logger.exception(f"Signal error: {e}")
        return None
