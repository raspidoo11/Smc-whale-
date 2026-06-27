import pandas as pd
import numpy as np
import logging
import joblib
from trade_manager import get_trade_history
import os
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_PATH = "models/xgboost_model.pkl"
os.makedirs("models", exist_ok=True)


def extract_features_from_trade(trade):
    """Extract features from closed trade for training"""
    features = {
        'volume_spike': trade.get('volume_spike', 0),
        'displacement': trade.get('displacement', 0),
        'trend_bull': 1 if trade.get('direction') == 'LONG' else 0,
        'sweep': trade.get('sweep', 0),
        'fvg': trade.get('fvg', 0),
        'atr': trade.get('atr', 0),
        'qty': trade.get('qty', 1),
        'risk_reward': abs((trade.get('tp', 0) - trade.get('entry', 0)) / (trade.get('entry', 0) - trade.get('sl', 0))) if trade.get('sl', 0) != trade.get('entry', 0) else 1.5,
    }
    return features


def train_model():
    """Train XGBoost on closed trades"""
    history = get_trade_history()
    if len(history) < 10:
        logger.info(f"Only {len(history)} trades. Waiting for more data...")
        return None

    logger.info(f"Training XGBoost on {len(history)} closed trades...")

    data = []
    for trade in history:
        if trade.get("status") in ["WIN", "LOSS"]:
            feat = extract_features_from_trade(trade)
            feat['target'] = 1 if trade["status"] == "WIN" else 0
            data.append(feat)

    if len(data) < 10:
        return None

    df = pd.DataFrame(data)
    X = df.drop('target', axis=1)
    y = df['target']

    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    logger.info(f"✅ XGBoost model trained! Historical accuracy: {model.score(X, y):.2f}")
    return model


def get_xgboost_probability(trade_features):
    """Get AI win probability for a new signal"""
    if not Path(MODEL_PATH).exists():
        return 50.0  # neutral until trained

    try:
        model = joblib.load(MODEL_PATH)
        X = pd.DataFrame([trade_features])
        prob = model.predict_proba(X)[0][1] * 100
        return round(prob, 1)
    except Exception as e:
        logger.error(f"XGBoost prediction failed: {e}")
        return 50.0
