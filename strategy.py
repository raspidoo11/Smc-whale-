import pandas as pd

def calculate_features(df):

```
df = df.copy()

df["atr"] = (
    df["high"] - df["low"]
).rolling(14).mean()

df["volume_ma"] = (
    df["volume"]
).rolling(20).mean()

df["volume_spike"] = (
    df["volume"] >
    df["volume_ma"] * 1.5
).astype(int)

df["body"] = abs(
    df["close"] - df["open"]
)

df["displacement"] = (
    df["body"] >
    df["atr"] * 0.7
).astype(int)

return df
```

def bullish_bos(df):

```
return (
    df["close"].iloc[-1] >
    df["high"].iloc[-10:-1].max()
)
```

def bearish_bos(df):

```
return (
    df["close"].iloc[-1] <
    df["low"].iloc[-10:-1].min()
)
```

def get_signal(df_15m, df_5m):

```
df_15m = calculate_features(df_15m)

df_5m = calculate_features(df_5m)

trend_bull = bullish_bos(df_15m)
trend_bear = bearish_bos(df_15m)

latest = df_5m.iloc[-1]

score = 0

if latest["volume_spike"] == 1:
    score += 35

if latest["displacement"] == 1:
    score += 35

if trend_bull:
    score += 15

if trend_bear:
    score += 15

entry = latest["close"]

atr = latest["atr"]

if trend_bull and score >= 70:

    sl = entry - atr

    tp = entry + (entry - sl) * 1.5

    return {
        "direction": "LONG",
        "confidence": score,
        "entry": entry,
        "sl": sl,
        "tp": tp
    }

if trend_bear and score >= 70:

    sl = entry + atr

    tp = entry - (sl - entry) * 1.5

    return {
        "direction": "SHORT",
        "confidence": score,
        "entry": entry,
        "sl": sl,
        "tp": tp
    }

return None
```
