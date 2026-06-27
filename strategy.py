import pandas as pd
import numpy as np

def calculate_features(df):

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


def bullish_bos(df):

    last_high = df["high"].iloc[-10:-1].max()

    return (
        df["close"].iloc[-1] > last_high
    )


def bearish_bos(df):

    last_low = df["low"].iloc[-10:-1].min()

    return (
        df["close"].iloc[-1] < last_low
    )


def confidence_score(df):

    score = 0

    latest = df.iloc[-1]

    if latest["volume_spike"] == 1:
        score += 25

    if latest["displacement"] == 1:
        score += 25

    if bullish_bos(df):
        score += 25

    if bearish_bos(df):
        score += 25

    return score


def get_signal(df):

    df = calculate_features(df)

    score = confidence_score(df)

    if bullish_bos(df) and score >= 75:

        return {
            "direction": "LONG",
            "confidence": score
        }

    if bearish_bos(df) and score >= 75:

        return {
            "direction": "SHORT",
            "confidence": score
        }

    return None
