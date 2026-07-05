import pandas as pd
import numpy as np
import logging
import joblib
import os
import json
from pathlib import Path
from datetime import datetime
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from trade_manager import get_trade_history
from config import MODELS_DIR, WIN_LABEL_MIN_R

logger = logging.getLogger(__name__)

# Paths derive from config.MODELS_DIR (default "data/models") instead of the
# old hardcoded "/app/data/models", which only existed on Railway and broke
# every local run on Windows/macOS.
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_model.pkl")
CHALLENGER_PATH = os.path.join(MODELS_DIR, "xgboost_model_challenger.pkl")
EXPECTED_R_MODEL_PATH = os.path.join(MODELS_DIR, "expected_r_model.pkl")
FEATURE_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")
# The expected-R model gets its OWN feature list. It used to reuse FEATURE_PATH
# (the classifier's list), but the two are saved at different times/conditions,
# so after a feature-set change they drift apart and XGBoost raises a
# feature_names mismatch at predict time. Separate files keep each model paired
# with the exact columns it was trained on.
EXPECTED_R_FEATURE_PATH = os.path.join(MODELS_DIR, "expected_r_feature_names.pkl")
FEATURE_HISTORY_PATH = os.path.join(MODELS_DIR, "feature_importance_history.json")
METRICS_HISTORY_PATH = os.path.join(MODELS_DIR, "training_metrics_history.json")
METADATA_PATH = os.path.join(MODELS_DIR, "training_metadata.json")
DIAGNOSTICS_PATH = os.path.join(MODELS_DIR, "diagnostics_report.json")
DIAGNOSTICS_STATE_PATH = os.path.join(MODELS_DIR, "diagnostics_state.json")

os.makedirs(MODELS_DIR, exist_ok=True)

ROLLING_WINDOW_SIZE = 220
RETRAIN_EVERY_N_TRADES = 5

# Backtest-backfilled trades (source == "backtest") train at reduced weight:
# they warm-start the model when real data is scarce, but a simulated fill
# must never outvote a real one.
BACKTEST_SAMPLE_WEIGHT = 0.5

MIN_TRADES_TO_TRAIN = 30
MIN_TRADES_FOR_ENSEMBLE = 90       # below this: logistic regression only, no ensemble
MIN_HOLDOUT_SIZE = 15
DIAGNOSTICS_EVERY_N_TRADES = 100
OVERFIT_GAP_ALERT_THRESHOLD = 0.20  # train AUC - holdout AUC beyond this gets logged as a warning

DROP_FEATURES = ["fvg_x_sweep"]  # duplicate of sweep_x_fvg


# ---------------------------------------------------------------------------
# Feature engineering (unchanged from phase14)
# ---------------------------------------------------------------------------

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

    # NOTE: qty_size and trade_duration_hours were removed here deliberately.
    # trade_duration_hours is target leakage — it's only known AFTER a trade
    # closes, but this classifier predicts at ENTRY time, so it can never be
    # supplied honestly at inference. qty_size isn't known at entry either
    # (quantity is sized downstream of the signal), so it was always a
    # constant default live while carrying the real value in training — a
    # train/serve skew. sl_tightness was just a duplicate of risk_pct.

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

    # Market-context features (persisted on the signal by strategy.py; all
    # None-safe so pre-upgrade history rows simply read as neutral).
    is_long = 1 if trade.get("direction") == "LONG" else -1
    features["funding_rate"] = float(trade.get("funding_rate") or 0.0)
    features["oi_change_pct"] = float(trade.get("oi_change_pct") or 0.0)
    features["btc_trend"] = float(trade.get("btc_trend") or 0.0)
    features["spread_pct"] = float(trade.get("spread_pct") or 0.0)
    features["symbol_win_rate"] = float(trade.get("symbol_win_rate") or 0.5)
    # Trading WITH BTC's structure vs against it — alts rarely win that fight.
    features["btc_aligned"] = 1 if features["btc_trend"] * is_long > 0 else 0
    # Positive funding on a long = paying to join the crowded side.
    features["funding_vs_direction"] = features["funding_rate"] * is_long

    # Sentiment regime: Fear & Greed index (0 fear .. 100 greed, 50 neutral).
    fng = float(trade.get("fng") or 50.0)
    features["fng"] = fng
    features["is_extreme_fear"] = 1 if fng <= 20 else 0
    features["is_extreme_greed"] = 1 if fng >= 80 else 0

    features["volume_x_displacement"] = trade.get("volume_spike", 0) * trade.get("displacement", 0)
    features["sweep_x_fvg"] = trade.get("sweep", 0) * trade.get("fvg", 0)
    features["volatility_x_risk"] = atr * features["risk_pct"]
    features["atr_x_confluence"] = atr * (confluence_score / 4.0)
    features["atr_x_session"] = atr * (1 if 7 <= hour <= 16 else 0)
    features["fvg_x_sweep"] = trade.get("fvg", 0) * trade.get("sweep", 0)  # duplicate, dropped before training
    features["rr_x_atr"] = features["risk_reward"] * atr

    return features


def calculate_historical_context(history):
    """Caller must pass only trades that occurred BEFORE the trade being featurized."""
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

    # `... or 0` guards against legacy rows where pnl was persisted as null
    # (from the pre-fix bug); float(None) would raise and crash training.
    cumulative_pnl = sum(float(t.get("pnl") or 0) for t in history[-20:])

    equity_curve = np.cumsum([float(t.get("pnl") or 0) for t in history[-50:]])
    if len(equity_curve) > 0:
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = running_max - equity_curve
        current_dd_pct = float(drawdowns[-1])
    else:
        current_dd_pct = 0

    return {
        "recent_win_rate": wins / max(len(recent), 1),
        "streak_count": streak,
        "cumulative_pnl": cumulative_pnl,
        "current_dd_pct": current_dd_pct,
    }


def calculate_realized_r(trade):
    """
    Spec item 2: Expected/realized R-multiple instead of only WIN/LOSS.
    R = how many multiples of planned risk were actually gained or lost.
    Positive for wins, negative for losses, magnitude reflects how close
    the trade got to its planned TP vs how much of its SL it ate.
    """
    entry = float(trade.get("entry") or 1)
    sl = float(trade.get("sl") or 0)
    exit_price = float(trade.get("exit_price") or entry)
    direction = trade.get("direction", "LONG")

    if direction == "LONG":
        risk_distance = abs(entry - sl)
        realized_distance = exit_price - entry
    else:
        risk_distance = abs(sl - entry)
        realized_distance = entry - exit_price

    if risk_distance <= 0:
        return 0.0

    return round(realized_distance / risk_distance, 4)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_json(path, default):
    if not Path(path).exists():
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Corrupted or unreadable JSON at {path} ({e}) — using default and backing up the bad file")
        try:
            corrupt_backup = f"{path}.corrupt.{int(datetime.utcnow().timestamp())}"
            os.replace(path, corrupt_backup)
        except OSError:
            pass
        return default


def save_json(path, data):
    """Atomic write: write to a temp file first, then rename into place.
    Prevents corruption if the process is killed/restarted mid-write."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_feature_history():
    return load_json(FEATURE_HISTORY_PATH, {})


def save_feature_history(history):
    save_json(FEATURE_HISTORY_PATH, history)


def load_metrics_history():
    return load_json(METRICS_HISTORY_PATH, [])


def save_metrics_history(metrics_history):
    save_json(METRICS_HISTORY_PATH, metrics_history[-200:])


def update_feature_importance_history(current_importance, history, decay_factor=0.85):
    for feature, importance in current_importance.items():
        if feature not in history:
            history[feature] = {"recent_importance": [], "decay_multiplier": 1.0}

        history[feature]["recent_importance"].append(float(importance))
        if len(history[feature]["recent_importance"]) > 10:
            history[feature]["recent_importance"].pop(0)

        avg_importance = np.mean(history[feature]["recent_importance"])

        if avg_importance < 0.01:
            history[feature]["decay_multiplier"] = max(0.3, history[feature]["decay_multiplier"] * decay_factor)
        else:
            history[feature]["decay_multiplier"] = min(1.0, history[feature]["decay_multiplier"] / decay_factor)

    return history


def get_decayed_features(feature_history, min_multiplier=0.4):
    return [f for f, data in feature_history.items() if data.get("decay_multiplier", 1.0) < min_multiplier]


# ---------------------------------------------------------------------------
# Feature frame construction
# ---------------------------------------------------------------------------

def build_feature_frame(history):
    """Walk-forward feature construction: context for trade i only uses trades before i."""
    rows = []
    for i, trade in enumerate(history):
        if trade.get("status") not in ["WIN", "LOSS"]:
            continue
        context = calculate_historical_context(history[:i])
        regime = trade.get("market_regime", "ranging")
        row = extract_pro_features_from_trade(trade, context, regime=regime)
        # Label engineering: positive class = "realized at least WIN_LABEL_MIN_R"
        # (default 0.5R), NOT "pnl > 0". A +0.05R scratch and a +2.5R runner
        # are different outcomes; labeling both WIN taught the model to
        # predict fee-noise. Balance/streak accounting still uses status.
        r = calculate_realized_r(trade)
        row["target"] = 1 if r >= WIN_LABEL_MIN_R else 0
        row["realized_r"] = r
        # Metadata, not a feature: string column, so prepare_X_y's
        # select_dtypes(number) drops it from X automatically. Used only to
        # down-weight simulated rows during fitting.
        row["sample_source"] = trade.get("source", "live")
        rows.append(row)

    return pd.DataFrame(rows)


def prepare_X_y(df, feature_history=None, drop_decayed=True, target_col="target"):
    drop_cols = [c for c in ("target", "realized_r") if c in df.columns]
    X = df.drop(columns=drop_cols)
    X = X.select_dtypes(include=[np.number]).fillna(0)

    cols_to_drop = [c for c in DROP_FEATURES if c in X.columns]
    if drop_decayed and feature_history:
        cols_to_drop += [c for c in get_decayed_features(feature_history) if c in X.columns]
    if cols_to_drop:
        X = X.drop(columns=list(set(cols_to_drop)))

    y = df[target_col]
    return X, y


def make_sample_weights(n, half_life=60):
    """Recency decay for LIVE incremental training (rolling ~220-trade window).
    half_life=None -> uniform weights: required for large offline corpora,
    where a 60-trade half-life would silently reduce thousands of trades to
    ~87 effective samples (sum of the geometric series) and defeat the point
    of building the corpus at all."""
    if half_life is None:
        return np.ones(n)
    ages = np.arange(n)[::-1]
    return 0.5 ** (ages / half_life)


def split_train_holdout(df, holdout_size):
    """
    Build a train/holdout split for the classifier.

    Historically this was a strict chronological split (train = everything
    except the last `holdout_size` rows). That's the "honest" walk-forward
    approach, but it has a failure mode: if the *training* slice happens to
    land on a streak (e.g. the bot's first 20+ closed trades were all WINs,
    or all LOSSes), `y_train.nunique() < 2` and training silently skips
    forever — even once the full window has both classes — because the
    fixed training slice never changes composition until the streak ages
    out of the rolling window.

    Fix: try a class-stratified split first, so both splits are guaranteed
    to contain both classes whenever the full window does. We keep each
    split sorted back into chronological order afterward so the recency
    sample-weighting in `make_sample_weights` still behaves sensibly.
    Falls back to the old chronological split if stratification isn't
    possible (e.g. a class has fewer members than needed for the split).
    """
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=holdout_size,
            stratify=df["target"],
            shuffle=True,
            random_state=42,
        )
        train_df = train_df.sort_index()
        test_df = test_df.sort_index()
        logger.info(
            f"   Split method: stratified (train={len(train_df)}, holdout={len(test_df)})"
        )
        return train_df, test_df
    except ValueError as e:
        logger.warning(
            f"Stratified split failed ({e}) — falling back to chronological split. "
            f"This can happen if one class has very few trades."
        )
        train_df = df.iloc[:-holdout_size]
        test_df = df.iloc[-holdout_size:]
        logger.info(
            f"   Split method: chronological fallback (train={len(train_df)}, holdout={len(test_df)})"
        )
        return train_df, test_df


# ---------------------------------------------------------------------------
# Ensemble (spec item 7 — kept to 2 models; CatBoost/RandomForest skipped,
# see conversation notes on why a 4-way ensemble isn't appropriate at this N)
# ---------------------------------------------------------------------------

class EnsembleClassifier:
    """Soft-voting ensemble over pre-calibrated classifiers. Must stay a
    module-level class so joblib can pickle/unpickle it."""

    def __init__(self, models, weights=None, names=None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.names = names or [f"model_{i}" for i in range(len(models))]

    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        for model, w in zip(self.models, self.weights):
            probs += w * model.predict_proba(X)
        return probs

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def evaluate_model(model, X_test, y_test):
    if len(X_test) == 0 or y_test.nunique() < 2:
        return None
    probs = model.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = None
    try:
        ll = log_loss(y_test, probs, labels=[0, 1])
    except ValueError:
        ll = None
    brier = brier_score_loss(y_test, probs)
    return {"auc": auc, "log_loss": ll, "brier": brier, "n_test": len(y_test)}


def fit_candidate_model(X_train, y_train, use_ensemble=True, source_weights=None,
                        recency_half_life=60):
    win_count = int((y_train == 1).sum())
    loss_count = int((y_train == 0).sum())
    weights = make_sample_weights(len(y_train), half_life=recency_half_life)
    if source_weights is not None:
        weights = weights * np.asarray(source_weights)
    cv_folds = 3 if len(y_train) >= 60 else 2

    if not use_ensemble:
        base = LogisticRegression(max_iter=1000, class_weight="balanced")
        base.fit(X_train, y_train, sample_weight=weights)
        try:
            calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=cv_folds)
            calibrated.fit(X_train, y_train, sample_weight=weights)
        except Exception as e:
            logger.warning(f"Calibration failed ({e}), using uncalibrated model")
            calibrated = base
        return calibrated, [("logistic_regression", base)], "logistic_regression"

    xgb_model = XGBClassifier(
        n_estimators=140, learning_rate=0.065, max_depth=6, min_child_weight=4,
        subsample=0.82, colsample_bytree=0.82, gamma=0.65, reg_alpha=0.55, reg_lambda=1.1,
        scale_pos_weight=max(loss_count / max(win_count, 1), 1),
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    xgb_model.fit(X_train, y_train, sample_weight=weights)

    lgb_model = LGBMClassifier(
        n_estimators=140, learning_rate=0.065, max_depth=6, min_child_samples=8,
        subsample=0.82, colsample_bytree=0.82, reg_alpha=0.55, reg_lambda=1.1,
        class_weight="balanced", random_state=42, verbosity=-1,
    )
    lgb_model.fit(X_train, y_train, sample_weight=weights)

    try:
        xgb_cal = CalibratedClassifierCV(xgb_model, method="sigmoid", cv=cv_folds)
        xgb_cal.fit(X_train, y_train, sample_weight=weights)
        lgb_cal = CalibratedClassifierCV(lgb_model, method="sigmoid", cv=cv_folds)
        lgb_cal.fit(X_train, y_train, sample_weight=weights)
    except Exception as e:
        logger.warning(f"Calibration failed ({e}), using uncalibrated ensemble members")
        xgb_cal, lgb_cal = xgb_model, lgb_model

    ensemble = EnsembleClassifier([xgb_cal, lgb_cal], weights=[0.55, 0.45], names=["xgboost", "lightgbm"])
    raw_models = [("xgboost", xgb_model), ("lightgbm", lgb_model)]
    return ensemble, raw_models, "xgboost+lightgbm_ensemble"


def fit_expected_r_model(X_train, r_train, source_weights=None):
    """Spec item 2 (regression side): auxiliary model predicting realized R.
    Logged/exposed separately — does not replace the win/loss classifier."""
    model = XGBRegressor(
        n_estimators=100, learning_rate=0.07, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, verbosity=0,
    )
    model.fit(X_train, r_train, sample_weight=source_weights)
    return model


# ---------------------------------------------------------------------------
# Self-diagnostics (spec item 17)
# ---------------------------------------------------------------------------

def _group_win_rate(df, col, min_count=3):
    if col not in df.columns:
        return {}
    grouped = df.groupby(col)["target"].agg(["mean", "count"])
    grouped = grouped[grouped["count"] >= min_count]
    return {str(k): {"win_rate": round(v["mean"], 3), "n": int(v["count"])} for k, v in grouped.iterrows()}


def generate_diagnostic_report(history, feature_importance=None):
    closed = [t for t in history if t.get("status") in ("WIN", "LOSS")]
    if len(closed) < 20:
        return None

    df = pd.DataFrame(closed)
    df["target"] = (df["status"] == "WIN").astype(int)
    df["hour"] = df.get("hour", 12)

    def session_bucket(h):
        h = int(h) if pd.notna(h) else 12
        if 7 <= h <= 11:
            return "london"
        if 12 <= h <= 16:
            return "ny"
        if h >= 22 or h <= 6:
            return "asian"
        if 17 <= h <= 21:
            return "quiet"
        return "overlap"

    df["session"] = df["hour"].apply(session_bucket)

    session_stats = _group_win_rate(df, "session")
    weekday_stats = _group_win_rate(df, "day_of_week")
    regime_stats = _group_win_rate(df, "market_regime")
    symbol_stats = _group_win_rate(df, "symbol", min_count=3)

    def best_worst(stats):
        if not stats:
            return None, None
        ranked = sorted(stats.items(), key=lambda kv: kv[1]["win_rate"])
        return ranked[-1][0], ranked[0][0]

    best_session, worst_session = best_worst(session_stats)
    best_weekday, worst_weekday = best_worst(weekday_stats)
    best_regime, worst_regime = best_worst(regime_stats)

    top_symbols, bottom_symbols = [], []
    if symbol_stats:
        ranked_symbols = sorted(symbol_stats.items(), key=lambda kv: kv[1]["win_rate"], reverse=True)
        top_symbols = ranked_symbols[:5]
        bottom_symbols = ranked_symbols[-5:]

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "n_trades": len(closed),
        "overall_win_rate": round(df["target"].mean(), 3),
        "session_stats": session_stats,
        "best_session": best_session,
        "worst_session": worst_session,
        "weekday_stats": weekday_stats,
        "best_weekday": best_weekday,
        "worst_weekday": worst_weekday,
        "regime_stats": regime_stats,
        "best_regime": best_regime,
        "worst_regime": worst_regime,
        "top_symbols": top_symbols,
        "bottom_symbols": bottom_symbols,
    }

    if feature_importance:
        ranked_features = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)
        report["best_features"] = ranked_features[:10]
        report["worst_features"] = ranked_features[-10:]

    save_json(DIAGNOSTICS_PATH, report)

    logger.info("\n   📋 Self-Diagnostic Report Generated")
    logger.info(f"      Overall Win Rate: {report['overall_win_rate']:.1%} over {report['n_trades']} trades")
    logger.info(f"      Best/Worst Session: {best_session} / {worst_session}")
    logger.info(f"      Best/Worst Regime: {best_regime} / {worst_regime}")

    return report


def maybe_run_diagnostics(history, feature_importance):
    state = load_json(DIAGNOSTICS_STATE_PATH, {"last_run_count": 0})
    closed_count = len([t for t in history if t.get("status") in ("WIN", "LOSS")])

    if closed_count - state.get("last_run_count", 0) >= DIAGNOSTICS_EVERY_N_TRADES:
        report = generate_diagnostic_report(history, feature_importance)
        if report:
            state["last_run_count"] = closed_count
            save_json(DIAGNOSTICS_STATE_PATH, state)
        return report
    return None


# ---------------------------------------------------------------------------
# Best-effort SHAP explanation (spec item 19). Never breaks training if the
# `shap` package isn't installed — logs a one-time notice instead.
# ---------------------------------------------------------------------------

_shap_warned = False


def explain_prediction(raw_model, X_row, top_n=5):
    global _shap_warned
    try:
        import shap
        explainer = shap.TreeExplainer(raw_model)
        shap_values = explainer.shap_values(X_row)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        contributions = dict(zip(X_row.columns, np.abs(shap_values[0])))
        ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_n]
    except ImportError:
        if not _shap_warned:
            logger.info("shap not installed — skipping feature-attribution logging (pip install shap to enable)")
            _shap_warned = True
        return None
    except Exception as e:
        logger.debug(f"SHAP explanation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Helper: log + verify that a joblib file actually landed on disk
# ---------------------------------------------------------------------------

def _dump_and_verify(obj, path, label):
    """joblib.dump() doesn't raise on most silent failures (e.g. a bad mount
    path that still resolves to a writable-but-wrong directory), so we
    explicitly re-check Path.exists() + file size after every save and log
    the outcome. This is what you should grep your logs for to confirm the
    model actually landed on disk."""
    try:
        # Atomic: dump to a temp file then os.replace() into place, so a crash
        # mid-write can't leave a half-written .pkl that joblib.load() chokes
        # on next inference. Mirrors the atomic JSON writes in trade_manager.
        tmp = f"{path}.tmp"
        joblib.dump(obj, tmp)
        os.replace(tmp, path)
        p = Path(path)
        if p.exists():
            size_kb = p.stat().st_size / 1024
            logger.info(f"   💾 Saved {label} → {path} ({size_kb:.1f} KB) ✅ confirmed on disk")
        else:
            logger.error(f"   ❌ {label} save reported no error, but {path} does NOT exist on disk!")
    except Exception as e:
        logger.error(f"   ❌ Failed to save {label} to {path}: {e}")
        raise


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_model_incremental(force_retrain=False):
    """
    Phase 15: walk-forward context + calibrated 2-model ensemble +
    champion/challenger promotion + expected-R regression + self-diagnostics
    + overfit detection + full metadata persistence.

    Phase 16 fix: stratified train/holdout split (see split_train_holdout)
    so an early WIN or LOSS streak can no longer permanently starve training
    of both classes. Also adds explicit "is training actually running /
    did the files really get written" logging throughout.
    """
    history = get_trade_history()
    logger.info(f"🚀 Retrain: {len(history)} history rows")

    if len(history) < MIN_TRADES_TO_TRAIN:
        logger.info(f"⏳ Waiting for trades ({len(history)}/{MIN_TRADES_TO_TRAIN}) — skipping this run")
        return None

    full_history_for_diagnostics = history  # diagnostics look at everything, not just the rolling window

    if len(history) > ROLLING_WINDOW_SIZE:
        history = history[-ROLLING_WINDOW_SIZE:]
        logger.info(f"📉 Rolling Window Active: Using last {ROLLING_WINDOW_SIZE} trades")

    df = build_feature_frame(history)
    logger.info(f"   Labeled (WIN/LOSS) trades available for training: {len(df)}")

    if len(df) < 20:
        logger.info(f"⏳ Not enough labeled trades to train ({len(df)}/20) — skipping this run")
        return None

    # Guard: if the ENTIRE window is single-class, there is genuinely nothing
    # to learn yet (this is different from the old bug, where a single-class
    # *slice* of an otherwise-mixed window blocked training forever).
    if df["target"].nunique() < 2:
        win_ct = int((df["target"] == 1).sum())
        loss_ct = int((df["target"] == 0).sum())
        logger.info(
            f"⏳ Skipping retrain: entire rolling window has only one class so far "
            f"(WIN={win_ct}, LOSS={loss_ct}). Need at least one of each to train."
        )
        return None

    holdout_size = min(MIN_HOLDOUT_SIZE, max(5, len(df) // 5))
    train_df, test_df = split_train_holdout(df, holdout_size)

    feature_history = load_feature_history()
    use_ensemble = len(train_df) >= MIN_TRADES_FOR_ENSEMBLE
    logger.info(f"   Model family: {'XGBoost+LightGBM ensemble' if use_ensemble else 'Logistic regression only'} "
                f"(train size={len(train_df)}, ensemble threshold={MIN_TRADES_FOR_ENSEMBLE})")

    X_train, y_train = prepare_X_y(train_df, feature_history)
    X_test, y_test = prepare_X_y(test_df, feature_history)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    if y_train.nunique() < 2:
        # Should be rare now that split_train_holdout stratifies, but keep
        # as a final safety net (e.g. a class with fewer members than holdout_size).
        win_ct = int((y_train == 1).sum())
        loss_ct = int((y_train == 0).sum())
        logger.info(
            f"⏳ Training data has only one class after split (WIN={win_ct}, LOSS={loss_ct}), "
            f"skipping retrain this run"
        )
        return None

    def _source_mults(frame):
        if "sample_source" not in frame.columns:
            return None
        return np.where(frame["sample_source"] == "backtest", BACKTEST_SAMPLE_WEIGHT, 1.0)

    logger.info("   🔧 Fitting challenger model on train split...")
    challenger, challenger_raw_models, model_type = fit_candidate_model(
        X_train, y_train, use_ensemble=use_ensemble, source_weights=_source_mults(train_df)
    )
    challenger_metrics = evaluate_model(challenger, X_test, y_test)
    logger.info(f"   Challenger holdout metrics: {challenger_metrics}")

    # Overfit detection (spec item 13): compare train-set AUC to honest holdout AUC.
    train_metrics = evaluate_model(challenger, X_train, y_train)
    if train_metrics and challenger_metrics and train_metrics.get("auc") and challenger_metrics.get("auc"):
        gap = train_metrics["auc"] - challenger_metrics["auc"]
        if gap > OVERFIT_GAP_ALERT_THRESHOLD:
            logger.warning(
                f"⚠️  Possible overfitting: train AUC {train_metrics['auc']:.3f} vs "
                f"holdout AUC {challenger_metrics['auc']:.3f} (gap {gap:.3f})"
            )

    # Refit on the full window for the model that actually gets promoted, but
    # the promotion decision itself is based on the honest holdout above.
    logger.info("   🔧 Refitting final model on full window...")
    X_full, y_full = prepare_X_y(df, feature_history)
    final_model, final_raw_models, _ = fit_candidate_model(
        X_full, y_full, use_ensemble=use_ensemble, source_weights=_source_mults(df)
    )

    # Expected-R auxiliary regression (spec item 2)
    df_r = df.loc[X_full.index] if len(df) == len(X_full) else df
    r_target = df["realized_r"].loc[X_full.index] if "realized_r" in df.columns else None
    if r_target is not None and r_target.notna().sum() >= 20:
        try:
            logger.info("   🔧 Fitting expected-R regression model...")
            r_model = fit_expected_r_model(X_full, r_target, source_weights=_source_mults(df))
            _dump_and_verify(r_model, EXPECTED_R_MODEL_PATH, "expected-R model")
            # Save the feature list this model was trained on, alongside it, so
            # get_expected_r always aligns X to the right columns.
            _dump_and_verify(X_full.columns.tolist(), EXPECTED_R_FEATURE_PATH, "expected-R feature list")
        except Exception as e:
            logger.warning(f"Expected-R model training failed: {e}")
    else:
        logger.info("   Skipping expected-R model (not enough realized_r data yet)")

    metrics_history = load_metrics_history()
    champion_metrics = None

    if Path(MODEL_PATH).exists() and not force_retrain:
        try:
            champion = joblib.load(MODEL_PATH)
            champion_features = joblib.load(FEATURE_PATH)
            X_test_champ = X_test.reindex(columns=champion_features, fill_value=0)
            champion_metrics = evaluate_model(champion, X_test_champ, y_test)
            logger.info(f"   Existing champion loaded from {MODEL_PATH}, holdout metrics: {champion_metrics}")
        except Exception as e:
            logger.warning(f"Could not evaluate current champion: {e}")
    else:
        logger.info(f"   No existing champion found at {MODEL_PATH} (or force_retrain=True)")

    promote = True
    reason = "no existing champion"

    if champion_metrics:
        if not challenger_metrics:
            # We couldn't score the challenger on the holdout (e.g. single-class
            # holdout). Never replace a validated champion with an unvalidated
            # challenger — previously `promote` stayed True here and the
            # challenger was promoted without any comparison.
            promote = False
            reason = "challenger not evaluable on holdout (single-class); keeping champion"
        else:
            champ_auc = champion_metrics.get("auc") or 0
            chal_auc = challenger_metrics.get("auc") or 0
            if chal_auc >= champ_auc - 0.01:
                promote = True
                reason = f"challenger AUC {chal_auc:.3f} >= champion AUC {champ_auc:.3f} (-0.01 margin)"
            else:
                promote = False
                reason = f"challenger AUC {chal_auc:.3f} < champion AUC {champ_auc:.3f}, keeping champion"

    if promote:
        _dump_and_verify(final_model, MODEL_PATH, "PROMOTED model")
        _dump_and_verify(X_full.columns.tolist(), FEATURE_PATH, "feature name list")
        logger.info(f"✅ MODEL PROMOTED: {reason}")
    else:
        _dump_and_verify(final_model, CHALLENGER_PATH, "challenger model (not promoted)")
        logger.info(f"⏸️  MODEL NOT PROMOTED (saved as challenger): {reason}")

    # Combined feature importance across ensemble members
    current_importance = {}
    for name, raw in final_raw_models:
        if hasattr(raw, "feature_importances_"):
            for feat, imp in zip(X_full.columns, raw.feature_importances_):
                current_importance[feat] = current_importance.get(feat, 0) + float(imp)
        elif hasattr(raw, "coef_"):
            for feat, imp in zip(X_full.columns, np.abs(raw.coef_[0])):
                current_importance[feat] = current_importance.get(feat, 0) + float(imp)

    if current_importance:
        # normalize by number of models contributing
        n_models = len(final_raw_models)
        current_importance = {k: v / n_models for k, v in current_importance.items()}
        feature_history = update_feature_importance_history(current_importance, feature_history)
        save_feature_history(feature_history)

    win_count = int((y_full == 1).sum())
    loss_count = int((y_full == 0).sum())
    win_rate = win_count / max(len(y_full), 1)

    n_backtest = int((df["sample_source"] == "backtest").sum()) if "sample_source" in df.columns else 0
    n_real = len(df) - n_backtest

    # Compact summary — one line per fact instead of the old multi-block dump
    # (Railway rate-limits at 500 logs/sec and was dropping messages).
    logger.info(f"   Model: {model_type} | learned {len(df)} trades "
                f"(W{win_count}/L{loss_count}, {win_rate:.0%}) | "
                f"mix {n_real} real + {n_backtest} backtest @{BACKTEST_SAMPLE_WEIGHT}x")
    if challenger_metrics:
        auc = challenger_metrics.get("auc")
        brier = challenger_metrics.get("brier")
        logger.info(f"   Holdout: AUC {auc if auc is None else f'{auc:.3f}'} | "
                    f"Brier {brier if brier is None else f'{brier:.3f}'}")

    if current_importance:
        ranked = sorted(current_importance.items(), key=lambda kv: kv[1], reverse=True)
        top5 = ", ".join(f"{feat} {imp:.2f}" for feat, imp in ranked[:5])
        logger.info(f"   📊 Top features: {top5}")

    if len(history) > 0:
        last_trade = history[-1]
        logger.info(
            f"   📈 Last trade: #{last_trade.get('trade_no', '?')} "
            f"{last_trade.get('symbol', '?')} {last_trade.get('direction', '?')} "
            f"{last_trade.get('status', '?')} pnl {last_trade.get('pnl', 0) or 0:+.2f} "
            f"R {calculate_realized_r(last_trade):+.2f}"
        )

    metrics_history.append({
        "timestamp": datetime.utcnow().isoformat(),
        "n_trades": len(df),
        "win_rate": win_rate,
        "model_type": model_type,
        "promoted": promote,
        "reason": reason,
        "challenger_metrics": challenger_metrics,
        "champion_metrics": champion_metrics,
        "train_metrics": train_metrics,
    })
    save_metrics_history(metrics_history)

    # Metadata persistence (spec item 18)
    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "model_type": model_type,
        "promoted": promote,
        "n_trades_total": len(history),
        "n_trades_used": len(df),
        "rolling_window_size": ROLLING_WINDOW_SIZE,
        "holdout_size": holdout_size,
        "feature_list": X_full.columns.tolist(),
        "n_features": len(X_full.columns),
        "dropped_features": list(set(DROP_FEATURES + get_decayed_features(feature_history))),
        "calibration_method": "sigmoid (Platt scaling) via CalibratedClassifierCV",
        "ensemble_members": [name for name, _ in final_raw_models],
        "holdout_metrics": challenger_metrics,
        "train_metrics": train_metrics,
        "has_expected_r_model": Path(EXPECTED_R_MODEL_PATH).exists(),
    }
    save_json(METADATA_PATH, metadata)
    logger.info(f"   💾 Metadata saved → {METADATA_PATH}")

    # Self-diagnostics (spec item 17) — runs against full trade history, not
    # just the rolling window, on a ~100-trade cadence.
    maybe_run_diagnostics(full_history_for_diagnostics, current_importance)

    logger.info("🏁 Retrain finished")

    return final_model if promote else (champion if champion_metrics else final_model)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_model_load_logged = False  # avoid spamming logs on every single inference call


def get_xgboost_probability(trade_features):
    """Return the calibrated probability (5–95%) that this setup realizes at
    least WIN_LABEL_MIN_R (default 0.5R) — i.e. a MEANINGFUL win, not a
    fee-noise scratch. See build_feature_frame's label engineering.

    The old version added a `(recent_win_rate - 0.5) * 4` "nudge" here — but
    recent_win_rate is ALREADY a model feature, so recent form was being counted
    twice (once by the model, once by this post-hoc bump). Removed: the model's
    output is already Platt-calibrated, so we just clamp it to a sane band.
    """
    global _model_load_logged
    try:
        if not Path(MODEL_PATH).exists():
            logger.warning(f"⚠️  No model file found at {MODEL_PATH} — returning neutral 50.0% probability. "
                            f"Model has not been trained/promoted yet.")
            return 50.0

        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURE_PATH)

        if not _model_load_logged:
            logger.info(f"✅ Model loaded successfully from {MODEL_PATH} "
                        f"({len(feature_names)} features) — AI inference is active")
            _model_load_logged = True

        X = pd.DataFrame([trade_features])
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]
        X = X.select_dtypes(include=[np.number]).fillna(0)

        prob = model.predict_proba(X)[0][1] * 100
        return round(max(5, min(95, prob)), 1)

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return 50.0


def get_expected_r(trade_features):
    """Spec item 2 (regression side): expected R-multiple for a candidate trade.

    Uses EXPECTED_R_FEATURE_PATH (the expected-R model's OWN feature list), not
    the classifier's FEATURE_PATH — reusing the latter caused a feature_names
    mismatch whenever the feature set changed. Returns None (filter skipped) if
    the model or its feature list isn't present yet, e.g. right after a
    feature-set change, until the next retrain writes them as a matched pair.
    """
    try:
        if not Path(EXPECTED_R_MODEL_PATH).exists() or not Path(EXPECTED_R_FEATURE_PATH).exists():
            logger.debug("Expected-R model or its feature list not found yet — skipping expected-R prediction")
            return None

        model = joblib.load(EXPECTED_R_MODEL_PATH)
        feature_names = joblib.load(EXPECTED_R_FEATURE_PATH)

        # reindex guarantees exactly the trained columns, in order, filling any
        # absent ones with 0 — so XGBoost's feature-name check always passes.
        X = pd.DataFrame([trade_features]).reindex(columns=feature_names, fill_value=0)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        return round(float(model.predict(X)[0]), 3)
    except Exception as e:
        logger.error(f"❌ Expected-R prediction failed: {e}")
        return None


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
