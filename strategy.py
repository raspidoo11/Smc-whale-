import pandas as pd
import logging
import os
from datetime import datetime
from trade_manager import get_trade_history
from xgboost_trainer import calculate_historical_context, get_xgboost_probability

logger = logging.getLogger(__name__)

USE_XGBOOST = os.getenv("USE_XGBOOST", "false").lower() == "true"


def calculate_features(df):
    df = df.copy()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["volume_ma"] = df["volume"].rolling(20).mean()

    # Use slightly stricter volume multiplier in SMC mode
    volume_mult = 1.4 if not USE_XGBOOST else 1.5
    df["volume_spike"] = (df["volume"] > df["volume_ma"] * volume_mult).astype(int)

    df["body"] = abs(df["close"] - df["open"])
    df["displacement"] = (df["body"] > df["atr"] * 0.7).astype(int)
    return df


def bullish_bos(df):
    swing_high = df["high"].iloc[-10:-1].max()

    if USE_XGBOOST:
        # Strict for AI mode
        return df["close"].iloc[-1] > swing_high

    # === STRICTER for SMC mode (avoid equal highs / weak breaks) ===
    return df["close"].iloc[-1] > swing_high * 1.0015


def bearish_bos(df):
    swing_low = df["low"].iloc[-10:-1].min()

    if USE_XGBOOST:
        return df["close"].iloc[-1] < swing_low

    # === STRICTER for SMC mode ===
    return df["close"].iloc[-1] < swing_low * 0.9985


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

        bull_sweep = (
            latest["low"] < df_5m["low"].iloc[-10:-1].min()
            and latest["close"] > latest["open"]
        )

        bear_sweep = (
            latest["high"] > df_5m["high"].iloc[-10:-1].max()
            and latest["close"] < latest["open"]
        )

        bull_fvg = df_5m["low"].iloc[-1] > df_5m["high"].iloc[-3]
        bear_fvg = df_5m["high"].iloc[-1] < df_5m["low"].iloc[-3]

        score = 0
        if latest["volume_spike"] == 1:
            score += 25
        if latest["displacement"] == 1:
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
        context = (
            calculate_historical_context(history)
            if len(history) >= 5
            else {
                "recent_win_rate": 0.5,
                "streak_count": 0,
                "cumulative_pnl": 0,
                "current_dd_pct": 0,
            }
        )

        # SL / TP - Fixed for SMC, dynamic only in AI mode
        if trend_bull:
            swing_low = df_5m["low"].iloc[-8:-1].min()
            sl = min(swing_low * 0.9995, entry - atr * 0.8)
            tp = entry + (entry - sl) * 1.5

        elif trend_bear:
            swing_high = df_5m["high"].iloc[-8:-1].max()
            sl = max(swing_high * 1.0005, entry + atr * 0.8)
            tp = entry - (sl - entry) * 1.5

        else:
            return None

        risk_pct = abs(entry - sl) / max(entry, 0.0001)
        reward_pct = abs(tp - entry) / max(entry, 0.0001)
        risk_reward = abs(tp - entry) / max(abs(entry - sl), 0.0001)

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
                "confluence_count": sum([
                    latest["volume_spike"],
                    latest["displacement"],
                    1 if (bull_sweep or bear_sweep) else 0,
                    1 if (bull_fvg or bear_fvg) else 0
                ]),
                "is_london_open": 1 if 7 <= hour <= 11 else 0,
                "is_ny_open": 1 if 12 <= hour <= 16 else 0,
                "is_asian": 1 if (hour >= 22 or hour <= 6) else 0,
                "is_overlap": 1 if 8 <= hour <= 11 else 0,
                "is_quiet_time": 1 if 17 <= hour <= 21 else 0,
                "is_monday": 1 if day_of_week == 0 else 0,
                "is_friday": 1 if day_of_week == 4 else 0,
                "risk_pct": risk_pct,
                "reward_pct": reward_pct,
                "recent_win_rate": context.get("recent_win_rate", 0.5),
                "streak_count": context.get("streak_count", 0),
                "is_hot_streak": 1 if context.get("streak_count", 0) > 0 else 0,
                "cumulative_pnl": context.get("cumulative_pnl", 0),
                "current_dd_pct": context.get("current_dd_pct", 0),
                "volume_x_displacement": latest["volume_spike"] * latest["displacement"],
                "sweep_x_fvg": (1 if (bull_sweep or bear_sweep) else 0) * (1 if (bull_fvg or bear_fvg) else 0),
                "volatility_x_risk": float(atr) * risk_pct,
            }
            ai_prob = get_xgboost_probability(trade_features)

        # Confidence calculation (same style as your good version)
        final_confidence = int((score * 0.4) + (ai_prob * 0.6))

        # ==================== STRICTER SMC RULES ====================
        if not USE_XGBOOST:
            # Require displacement on the 5m entry candle
            if latest["displacement"] != 1:
                return None

            # Raise threshold in SMC mode for quality
            if final_confidence < 52:
                return None

        logger.info(
            f"Signal | Mode={'AI' if USE_XGBOOST else 'SMC_STRICT'} | "
            f"bull={trend_bull} | bear={trend_bear} | "
            f"score={score} | AI={ai_prob:.1f} | conf={final_confidence}"
        )

        if trend_bull and final_confidence >= 40:
            return {
                "direction": "LONG",
                "confidence": final_confidence,
                "entry": entry,
                "sl": float(sl),
                "tp": float(tp),
                "ai_prob": ai_prob,
                "volume_spike": int(latest["volume_spike"]),
                "displacement": int(latest["displacement"]),
                "sweep": 1 if bull_sweep else 0,
                "fvg": 1 if bull_fvg else 0,
            }

        if trend_bear and final_confidence >= 40:
            return {
                "direction": "SHORT",
                "confidence": final_confidence,
                "entry": entry,
                "sl": float(sl),
                "tp": float(tp),
                "ai_prob": ai_prob,
                "volume_spike": int(latest["volume_spike"]),
                "displacement": int(latest["displacement"]),
                "sweep": 1 if bear_sweep else 0,
                "fvg": 1 if bear_fvg else 0,
            }

        return None

    except Exception as e:
        logger.exception(f"Signal error: {e}")
        return None
