import pandas as pd
import logging
import os
from xgboost_trainer import get_xgboost_probability
from telegram_alerts import send_alert

logger = logging.getLogger(__name__)

USE_XGBOOST = os.getenv("USE_XGBOOST", "false").lower() == "true"


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

        # Volume Filter (stronger filter)
        if latest["volume_spike"] == 0 and latest["displacement"] == 0:
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
            score += 20
        if trend_bull:
            score += 15
        if trend_bear:
            score += 15
        if bull_sweep or bear_sweep:
            score += 15
        if bull_fvg or bear_fvg:
            score += 10

        entry = latest["close"]

        # XGBoost
        ai_prob = 50.0
        if USE_XGBOOST:
            trade_features = {
                'volume_spike': latest["volume_spike"],
                'displacement': latest["displacement"],
                'trend_bull': 1 if trend_bull else 0,
                'sweep': 1 if (bull_sweep or bear_sweep) else 0,
                'fvg': 1 if (bull_fvg or bear_fvg) else 0,
                'atr': float(atr),
                'qty': 1.0,
                'risk_reward': 1.5
            }
            ai_prob = get_xgboost_probability(trade_features)

            await send_alert(
                f"🤖 XGBoost Active\n\n"
                f"AI Win Prob: {ai_prob}%\n"
                f"Use this for better signal selection"
            )

        final_confidence = int(0.6 * score + 0.4 * ai_prob)

        logger.info(f"Signal check | trend_bull={trend_bull} | SMC_score={score} | AI={ai_prob} | Final={final_confidence} | XGBoost={'ON' if USE_XGBOOST else 'OFF'}")

        if trend_bull and final_confidence >= 50:
            sl = entry - (atr * 1.5)   # Wider SL
            tp = entry + (entry - sl) * 2.0   # 2:1 RR
            return {
                "direction": "LONG",
                "confidence": final_confidence,
                "entry": float(entry),
                "sl": float(sl),
                "tp": float(tp),
                "ai_prob": ai_prob
            }

        if trend_bear and final_confidence >= 50:
            sl = entry + (atr * 1.5)
            tp = entry - (sl - entry) * 2.0
            return {
                "direction": "SHORT",
                "confidence": final_confidence,
                "entry": float(entry),
                "sl": float(sl),
                "tp": float(tp),
                "ai_prob": ai_prob
            }

        return None

    except Exception as e:
        logger.exception(f"Signal error: {e}")
        return None
