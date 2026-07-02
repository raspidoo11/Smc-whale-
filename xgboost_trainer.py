import pandas as pd
import numpy as np
import logging
import joblib
import os
import json
from pathlib import Path
from datetime import datetime
from xgboost import XGBClassifier
from trade_manager import get_trade_history

logger = logging.getLogger(__name__)

MODEL_PATH = "/app/data/models/xgboost_model.pkl"
FEATURE_PATH = "/app/data/models/feature_names.pkl"
FEATURE_HISTORY_PATH = "/app/data/models/feature_importance_history.json"

os.makedirs("/app/data/models", exist_ok=True)

ROLLING_WINDOW_SIZE = 220          # Use last ~220 trades
RETRAIN_EVERY_N_TRADES = 5         # Retrain after every 5 new closed trades


def calculate_atr_percentile(atr_series, current_atr, window=50):
    if len(atr_series) < window:
        return 50.0
    recent_atr = atr_series.tail(window)
    percentile = (recent_atr < current_atr).mean() * 100
    return round(percentile, 1)


def detect_market_regime(df, window=30):
    if len(df) < window:
        return "ranging"
    atr = df["atr"].iloc[-1]
    atr_avg = df["atr"].tail(window).mean()
    price_change = abs(df["close"].iloc[-1] - df["close"].iloc[-window]) / df["close"].iloc[-window]

    if atr > atr_avg * 1.4:
        return "volatile"
    elif price_change > 0.04:
        return "trending"
    else:
        return "ranging"


def extract_pro_features_from_trade(trade, historical_context=None, regime="ranging"):
    features = {
        "volume_spike": trade.get("volume_spike", 0),
        "displacement": trade.get("displacement", 0),
        "trend_bull": 1 if trade.get("direction") == "LONG" else 0,
        "sweep": trade.get("sweep", 0),
        "fvg": trade.get("fvg", 0),
    }

    entry = float(trade.get("entry", 1))
    sl = float(trade.get("sl", 0))
    tp = float(trade.get("tp", 0))

    if trade.get("direction") == "LONG":
        risk_distance = abs(entry - sl)
        reward_distance = abs(tp - entry)
    else:
        risk_distance = abs(sl - entry)
        reward_distance = abs(entry - tp)

    features["risk_reward"] = reward_distance / max(risk_distance, 0.0001)
    features["risk_pct"] = risk_distance / max(entry, 0.0001)
    features["reward_pct"] = reward_distance / max(entry, 0.0001)
    features["adversity_ratio"] = risk_distance / max(reward_distance, 0.0001)

    atr = float(trade.get("atr", 0))
    body = float(trade.get("body", 0))
    volume = float(trade.get("volume", 0))
    volume_ma = float(trade.get("volume_ma", 1))

    features["body_ratio"] = body / max(atr, 0.0001)
    features["volume_strength"] = volume / max(volume_ma, 0.0001)
    features["atr_expansion"] = atr

    atr_percentile = trade.get("atr_percentile", 50.0)
    features["atr_percentile"] = atr_percentile
    features["is_high_volatility"] = 1 if atr_percentile > 70 else 0
    features["is_low_volatility"] = 1 if atr_percentile < 30 else 0

    features["regime_trending"] = 1 if regime == "trending" else 0
    features["regime_ranging"] = 1 if regime == "ranging" else 0
    features["regime_volatile"] = 1 if regime == "volatile" else 0

    features["distance_to_prev_high"] = trade.get("distance_to_prev_high", 0)
    features["distance_to_prev_low"] = trade.get("distance_to_prev_low", 0)
    features["distance_to_ema20"] = trade.get("distance_to_ema20", 0)
    features["distance_to_ema50"] = trade.get("distance_to_ema50", 0)
    features["distance_to_vwap"] = trade.get("distance_to_vwap", 0)

    confluence_score = (
        int(trade.get("volume_spike", 0))
        + int(trade.get("displacement", 0))
        + int(trade.get("sweep", 0))
        + int(trade.get("fvg", 0))
    )

    features["confluence_count"] = confluence_score
    features["confluence_strength"] = confluence_score / 4.0

    hour = int(trade.get("hour", 12))
    day = int(trade.get("day_of_week", 2))

    features["hour"] = hour
    features["day_of_week"] = day

    features["is_london_open"] = 1 if 7 <= hour <= 11 else 0
    features["is_ny_open"] = 1 if 12 <= hour <= 16 else 0
    features["is_asian"] = 1 if (hour >= 22 or hour <= 6) else 0
    features["is_overlap"] = 1 if 8 <= hour <= 11 else 0
    features["is_quiet_time"] = 1 if 17 <= hour <= 21 else 0

    features["is_monday"] = 1 if day == 0 else 0
    features["is_friday"] = 1 if day == 4 else 0

    rr = features["risk_reward"]
    features["is_scalp"] = 1 if rr < 1.8 else 0
    features["is_swing"] = 1 if rr >= 2.0 else 0

    features["qty_size"] = np.log1p(float(trade.get("qty", 1)))
    features["trade_duration_hours"] = float(trade.get("duration_hours", 1))
    features["sl_tightness"] = features["risk_pct"]

    if historical_context:
        features["recent_win_rate"] = historical_context.get("recent_win_rate", 0.5)
        features["streak_count"] = historical_context.get("streak_count", 0)
        features["is_hot_streak"] = 1 if historical_context.get("streak_count", 0) > 0 else 0
        features["cumulative_pnl"] = historical_context.get("cumulative_pnl", 0)
        features["current_dd_pct"] = historical_context.get("current_dd_pct", 0)
    else:
        features["recent_win_rate"] = 0.5
        features["streak_count"] = 0
        features["is_hot_streak"] = 0
        features["cumulative_pnl"] = 0
        features["current_dd_pct"] = 0

    features["volume_x_displacement"] = trade.get("volume_spike", 0) * trade.get("displacement", 0)
    features["sweep_x_fvg"] = trade.get("sweep", 0) * trade.get("fvg", 0)
    features["volatility_x_risk"] = atr * features["risk_pct"]
    features["atr_x_confluence"] = atr * (confluence_score / 4.0)
    features["atr_x_session"] = atr * (1 if 7 <= hour <= 16 else 0)
    features["fvg_x_sweep"] = trade.get("fvg", 0) * trade.get("sweep", 0)
    features["rr_x_atr"] = features["risk_reward"] * atr

    return features


def calculate_historical_context(history):
    if len(history) < 5:
        return {
            "recent_win_rate": 0.5,
            "streak_count": 0,
            "cumulative_pnl": 0,
            "current_dd_pct": 0,
        }

    recent = history[-10:]
    wins = sum(1 for t in recent if t.get("status") == "WIN")

    streak = 0
    streak_sign = None

    for trade in reversed(history):
        is_win = trade.get("status") == "WIN"
        if streak_sign is None:
            streak_sign = is_win
            streak = 1
        elif is_win == streak_sign:
            streak += 1
        else:
            break

    if streak_sign is False:
        streak = -streak

    cumulative_pnl = sum(float(t.get("pnl", 0)) for t in history[-20:])

    return {
        "recent_win_rate": wins / max(len(recent), 1),
        "streak_count": streak,
        "cumulative_pnl": cumulative_pnl,
        "current_dd_pct": 0,
    }


def load_feature_history():
    if Path(FEATURE_HISTORY_PATH).exists():
        with open(FEATURE_HISTORY_PATH, "r") as f:
            return json.load(f)
    return {}


def save_feature_history(history):
    with open(FEATURE_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def update_feature_importance_history(current_importance, history, decay_factor=0.85):
    for feature, importance in current_importance.items():
        if feature not in history:
            history[feature] = {"recent_importance": [], "decay_multiplier": 1.0}

        history[feature]["recent_importance"].append(importance)
        if len(history[feature]["recent_importance"]) > 10:
            history[feature]["recent_importance"].pop(0)

        avg_importance = np.mean(history[feature]["recent_importance"])

        if avg_importance < 0.01:
            history[feature]["decay_multiplier"] = max(0.3, history[feature]["decay_multiplier"] * decay_factor)
        else:
            history[feature]["decay_multiplier"] = min(1.0, history[feature]["decay_multiplier"] / decay_factor)

    return history


def train_model_incremental(force_retrain=False):
    """
    Phase 12/13: Online-style learning with rolling window + frequent retraining
    """
    history = get_trade_history()

    if len(history) < 30:
        logger.info(f"Waiting for trades ({len(history)}/30)")
        return None

    # Use rolling window
    if len(history) > ROLLING_WINDOW_SIZE:
        history = history[-ROLLING_WINDOW_SIZE:]
        logger.info(f"📉 Rolling Window Active: Using last {ROLLING_WINDOW_SIZE} trades")

    context = calculate_historical_context(history[:-1])

    rows = []
    for trade in history:
        if trade.get("status") not in ["WIN", "LOSS"]:
            continue
        regime = trade.get("market_regime", "ranging")
        row = extract_pro_features_from_trade(trade, context, regime=regime)
        row["target"] = 1 if trade["status"] == "WIN" else 0
        rows.append(row)

    if len(rows) < 20:
        return None

    df = pd.DataFrame(rows)
    X = df.drop(columns=["target"])
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = df["target"]

    win_count = int((y == 1).sum())
    loss_count = int((y == 0).sum())
    win_rate = win_count / max(len(y), 1)

    feature_history = load_feature_history()

    model = XGBClassifier(
        n_estimators=140,
        learning_rate=0.065,
        max_depth=6,
        min_child_weight=4,
        subsample=0.82,
        colsample_bytree=0.82,
        gamma=0.65,
        reg_alpha=0.55,
        reg_lambda=1.1,
        scale_pos_weight=max(loss_count / max(win_count, 1), 1),
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )

    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(X.columns.tolist(), FEATURE_PATH)

    current_importance = dict(zip(X.columns, model.feature_importances_))
    feature_history = update_feature_importance_history(current_importance, feature_history)
    save_feature_history(feature_history)

    train_accuracy = model.score(X, y)

    logger.info(f"\n✅ MODEL UPDATED (Online/Rolling Mode)!")
    logger.info(f"   Using rolling window of last {min(len(history), ROLLING_WINDOW_SIZE)} trades")
    logger.info(f"   Trades Learned: {len(rows)} (W: {win_count}, L: {loss_count})")
    logger.info(f"   Win Rate: {win_rate:.1%}")
    logger.info(f"   Train Accuracy: {train_accuracy:.3f}")

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info(f"\n   📊 Top 10 Features:")
    for idx, row in feature_importance.head(10).iterrows():
        decay = feature_history.get(row['feature'], {}).get('decay_multiplier', 1.0)
        logger.info(f"      {row['feature']}: {row['importance']:.3f} (decay: {decay:.2f})")

    if len(history) > 0:
        last_trade = history[-1]
        logger.info(f"\n   📈 Latest Trade Analysis:")
        logger.info(f"      Trade #{last_trade.get('trade_no', '?')}: {last_trade.get('symbol', '?')}")
        logger.info(f"      Direction: {last_trade.get('direction', '?')}")
        logger.info(f"      Entry → Exit: ${last_trade.get('entry', 0):.6f} → ${last_trade.get('exit_price', 0):.6f}")
        logger.info(f"      Result: {last_trade.get('status', '?')} ({last_trade.get('pnl', 0):+.2f}%)")

    return model


def get_xgboost_probability(trade_features, recent_win_rate=0.5):
    try:
        if not Path(MODEL_PATH).exists():
            return 50.0

        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURE_PATH)

        X = pd.DataFrame([trade_features])
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0

        X = X[feature_names]
        X = X.select_dtypes(include=[np.number]).fillna(0)

        raw_prob = model.predict_proba(X)[0][1] * 100
        performance_adjustment = (recent_win_rate - 0.5) * 8
        calibrated_prob = raw_prob + performance_adjustment
        calibrated_prob = max(5, min(95, calibrated_prob))

        return round(calibrated_prob, 1)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 50.0


def get_dynamic_confidence_threshold(regime="ranging", atr_percentile=50, recent_win_rate=0.5):
    base_threshold = 45

    if regime == "trending":
        base_threshold -= 5
    elif regime == "volatile":
        base_threshold += 10
    elif regime == "ranging":
        base_threshold += 6

    if recent_win_rate > 0.55:
        base_threshold -= 6
    elif recent_win_rate < 0.40:
        base_threshold += 12

    if atr_percentile > 80:
        base_threshold += 6
    elif atr_percentile < 20:
        base_threshold -= 4

    return max(30, min(75, base_threshold))


def get_ai_risk_percent(ai_prob, recent_drawdown=0.0, regime="ranging"):
    base_risk = 0.5

    if ai_prob >= 80:
        risk = base_risk * 2.0
    elif ai_prob >= 70:
        risk = base_risk * 1.6
    elif ai_prob >= 60:
        risk = base_risk * 1.2
    else:
        risk = base_risk * 0.6

    if recent_drawdown > 5:
        risk *= 0.6
    elif recent_drawdown > 10:
        risk *= 0.4

    if regime == "volatile":
        risk *= 0.75
    elif regime == "trending":
        risk *= 1.15

    return round(max(0.2, min(2.5, risk)), 2)
