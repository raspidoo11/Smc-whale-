import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

USE_XGBOOST = os.getenv("USE_XGBOOST", "false").lower() == "true"

try:
    from xgboost_continuous_learning import get_xgboost_probability
except ImportError:
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

# More aggressive BOS
def bullish_bos(df):
    recent_high = df["high"].iloc[-15:-1].max()
    return df["close"].iloc[-1] > recent_high * 0.999

def bearish_bos(df):
    recent_low = df["low"].iloc[-15:-1].min()
    return df["close"].iloc[-1] < recent_low * 1.001

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
            latest["low"] < df_5m["low"].iloc[-12:-1].min()
            and latest["close"] > latest["open"]
        )
        bear_sweep = (
            latest["high"] > df_5m["high"].iloc[-12:-1].max()
            and latest["close"] < latest["open"]
        )
        
        bull_fvg = df_5m["low"].iloc[-1] > df_5m["high"].iloc[-3]
        bear_fvg = df_5m["high"].iloc[-1] < df_5m["low"].iloc[-3]
        
        score = 0
        if latest["volume_spike"] == 1:
            score += 25
        if latest["displacement"] == 1:
            score += 25   # Boosted for early momentum
        if trend_bull:
            score += 15
        if trend_bear:
            score += 15
        if bull_sweep or bear_sweep:
            score += 20   # Boosted
        if bull_fvg or bear_fvg:
            score += 10
        
        entry = latest["close"]
        
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
                'risk_reward': 1.5,
                'hour': 12,
                'day_of_week': 2,
                'confidence': int(0.6 * score + 0.4 * 50),
                'ai_prob': 50.0,
                'body_ratio': latest["body"] / max(atr, 0.0001),
                'volume_strength': latest["volume"] / max(latest["volume_ma"], 0.0001),
                'atr_expansion': float(atr),
                'confluence_count': sum([latest["volume_spike"], latest["displacement"], bull_sweep or bear_sweep, bull_fvg or bear_fvg]),
                'is_london_open': 0,
                'is_ny_open': 0,
                'is_asian': 0,
                'is_overlap': 0,
                'is_quiet_time': 0,
                'is_monday': 0,
                'is_friday': 0,
                'is_scalp': 0,
                'is_swing': 0,
                'qty_size': 0.0,
                'trade_duration_hours': 1,
                'sl_tightness': 0.01,
                'recent_win_rate': 0.5,
                'streak_count': 0,
                'is_hot_streak': 0,
                'cumulative_pnl': 0,
                'current_dd_pct': 0,
                'volume_x_displacement': latest["volume_spike"] * latest["displacement"],
                'confluence_x_confidence': 0,
                'sweep_x_fvg': (1 if (bull_sweep or bear_sweep) else 0) * (1 if (bull_fvg or bear_fvg) else 0),
                'volatility_x_risk': 0,
                'risk_pct': 0.01,
                'reward_pct': 0.015,
                'adversity_ratio': 0.67,
                'smc_ai_divergence': 0
            }
            ai_prob = get_xgboost_probability(trade_features)
        
        final_confidence = int(0.6 * score + 0.4 * ai_prob)
        
        logger.info(
            f"Signal check | trend_bull={trend_bull} | trend_bear={trend_bear} | "
            f"SMC_score={score} | AI={ai_prob:.1f} | Final={final_confidence} | "
            f"XGBoost={'ON' if USE_XGBOOST else 'OFF'}"
        )
        
        # Aggressive entry logic
        if (trend_bull or latest["displacement"] == 1) and final_confidence >= 40:
            sl = entry - atr * 1.1
            tp = entry + (entry - sl) * 1.5
            
            return {
                "direction": "LONG",
                "confidence": final_confidence,
                "entry": float(entry),
                "sl": float(sl),
                "tp": float(tp),
                "ai_prob": ai_prob,
                "volume_spike": latest["volume_spike"],
                "displacement": latest["displacement"],
                "sweep": 1 if (bull_sweep or bear_sweep) else 0,
                "fvg": 1 if (bull_fvg or bear_fvg) else 0
            }
        
        if (trend_bear or latest["displacement"] == 1) and final_confidence >= 40:
            sl = entry + atr * 1.1
            tp = entry - (sl - entry) * 1.5
            
            return {
                "direction": "SHORT",
                "confidence": final_confidence,
                "entry": float(entry),
                "sl": float(sl),
                "tp": float(tp),
                "ai_prob": ai_prob,
                "volume_spike": latest["volume_spike"],
                "displacement": latest["displacement"],
                "sweep": 1 if (bull_sweep or bear_sweep) else 0,
                "fvg": 1 if (bull_fvg or bear_fvg) else 0
            }
        
        return None
    
    except Exception as e:
        logger.exception(f"Signal error: {e}")
        return None
