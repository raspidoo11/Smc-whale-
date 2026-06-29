import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

USE_XGBOOST = os.getenv("USE_XGBOOST", "false").lower() == "true"

try:
    from xgboost_trainer import get_xgboost_probability
except ImportError:
    def get_xgboost_probability(features):
        return 50.0

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
        
        bull_sweep = latest["low"] < df_5m["low"].iloc[-10:-1].min() and latest["close"] > latest["open"]
        bear_sweep = latest["high"] > df_5m["high"].iloc[-10:-1].max() and latest["close"] < latest["open"]
        
        bull_fvg = df_5m["low"].iloc[-1] > df_5m["high"].iloc[-3]
        bear_fvg = df_5m["high"].iloc[-1] < df_5m["low"].iloc[-3]
        
        score = 0
        if latest["volume_spike"] == 1: score += 25
        if latest["displacement"] == 1: score += 25
        if trend_bull: score += 20
        if trend_bear: score += 20
        if bull_sweep or bear_sweep: score += 20
        if bull_fvg or bear_fvg: score += 15
        
        entry = latest["close"]
        
        ai_prob = 50.0
        if USE_XGBOOST:
            trade_features = { ... }  # your existing features dict
            ai_prob = get_xgboost_probability(trade_features)
        
        final_confidence = int(0.4 * score + 0.6 * ai_prob)   # AI 60%
        
        logger.info(f"Signal check | SMC={score} | AI={ai_prob:.1f} | Final={final_confidence}")
        
        if (trend_bull or latest["displacement"] == 1) and final_confidence >= 40:
            swing_low = df_5m["low"].iloc[-8:-1].min()
            sl = min(swing_low * 0.9995, entry - atr * 0.8)
            tp = entry + (entry - sl) * 1.5
            return {"direction": "LONG", "confidence": final_confidence, "entry": float(entry), "sl": float(sl), "tp": float(tp), "ai_prob": ai_prob, ...}
        
        if (trend_bear or latest["displacement"] == 1) and final_confidence >= 40:
            swing_high = df_5m["high"].iloc[-8:-1].max()
            sl = max(swing_high * 1.0005, entry + atr * 0.8)
            tp = entry - (sl - entry) * 1.5
            return {"direction": "SHORT", "confidence": final_confidence, "entry": float(entry), "sl": float(sl), "tp": float(tp), "ai_prob": ai_prob, ...}
        
        return None
    
    except Exception as e:
        logger.exception(f"Signal error: {e}")
        return None
