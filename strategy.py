import pandas as pd


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

        # Additional confluences
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

        if trend_bull and score >= 60:
            sl = entry - atr
            tp = entry + (entry - sl) * 1.5
            return {
                "direction": "LONG",
                "confidence": min(99, score),
                "entry": float(entry),
                "sl": float(sl),
                "tp": float(tp)
            }

        if trend_bear and score >= 60:
            sl = entry + atr
            tp = entry - (sl - entry) * 1.5
            return {
                "direction": "SHORT",
                "confidence": min(99, score),
                "entry": float(entry),
                "sl": float(sl),
                "tp": float(tp)
            }

        return None

    except Exception:
        return None
