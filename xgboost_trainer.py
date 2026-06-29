import pandas as pd
import numpy as np
import logging
import joblib
from trade_manager import get_trade_history
import os
from pathlib import Path
from xgboost import XGBClassifier
from datetime import datetime

logger = logging.getLogger(__name__)

MODEL_PATH = "models/xgboost_model.pkl"
TRADE_ANALYSIS_PATH = "models/trade_analysis.json"
os.makedirs("models", exist_ok=True)


def extract_pro_features_from_trade(trade, historical_context=None):
    """Professional feature engineering for pro trader thinking"""

    features = {
        'volume_spike': trade.get('volume_spike', 0),
        'displacement': trade.get('displacement', 0),
        'trend_bull': 1 if trade.get('direction') == 'LONG' else 0,
        'sweep': trade.get('sweep', 0),
        'fvg': trade.get('fvg', 0),
    }

    entry = trade.get('entry', 1)
    sl = trade.get('sl', 0)
    tp = trade.get('tp', 0)
    
    if trade.get('direction') == 'LONG':
        risk_distance = entry - sl
        reward_distance = tp - entry
        adversity_ratio = risk_distance / max(reward_distance, 0.0001)
    else:
        risk_distance = sl - entry
        reward_distance = entry - tp
        adversity_ratio = risk_distance / max(reward_distance, 0.0001)
    
    features['adversity_ratio'] = adversity_ratio
    features['risk_reward'] = trade.get('risk_reward', 1.5)
    features['risk_pct'] = abs(entry - sl) / entry if entry != 0 else 0
    features['reward_pct'] = abs(tp - entry) / entry if entry != 0 else 0
    features['body_ratio'] = trade.get('body', 0) / max(trade.get('atr', 1), 0.0001)
    features['volume_strength'] = trade.get('volume', 0) / max(trade.get('volume_ma', 1), 0.0001)
    features['atr_expansion'] = trade.get('atr', 0)

    confluence_score = 0
    if trade.get('volume_spike', 0): confluence_score += 1
    if trade.get('displacement', 0): confluence_score += 1
    if trade.get('sweep', 0): confluence_score += 1
    if trade.get('fvg', 0): confluence_score += 1
    features['confluence_count'] = confluence_score

    hour = trade.get('hour', 12)
    features['hour'] = hour
    features['is_london_open'] = 1 if 7 <= hour <= 11 else 0
    features['is_ny_open'] = 1 if 12 <= hour <= 16 else 0
    features['is_asian'] = 1 if (hour >= 22 or hour <= 6) else 0
    features['is_overlap'] = 1 if 8 <= hour <= 11 else 0
    features['is_quiet_time'] = 1 if 17 <= hour <= 21 else 0

    day = trade.get('day_of_week', 2)
    features['day_of_week'] = day
    features['is_monday'] = 1 if day == 0 else 0
    features['is_friday'] = 1 if day == 4 else 0

    features['confidence'] = trade.get('confidence', 50)
    features['ai_prob'] = trade.get('ai_prob', 50)
    features['smc_ai_divergence'] = abs(trade.get('confidence', 50) - trade.get('ai_prob', 50))

    features['is_scalp'] = 1 if trade.get('risk_reward', 1.5) < 1.8 else 0
    features['is_swing'] = 1 if trade.get('risk_reward', 1.5) >= 2.0 else 0
    features['qty_size'] = np.log1p(trade.get('qty', 1))
    features['trade_duration_hours'] = trade.get('duration_hours', 1)
    features['sl_tightness'] = features['risk_pct']

    if historical_context:
        features['recent_win_rate'] = historical_context.get('recent_win_rate', 0.5)
        features['streak_count'] = historical_context.get('streak_count', 0)
        features['is_hot_streak'] = 1 if historical_context.get('streak_count', 0) > 0 else 0
        features['cumulative_pnl'] = np.log1p(max(historical_context.get('cumulative_pnl', 0), 0.01))
        features['current_dd_pct'] = historical_context.get('current_dd_pct', 0)
    else:
        features['recent_win_rate'] = 0.5
        features['streak_count'] = 0
        features['is_hot_streak'] = 0
        features['cumulative_pnl'] = 0
        features['current_dd_pct'] = 0

    features['volume_x_displacement'] = trade.get('volume_spike', 0) * trade.get('displacement', 0)
    features['confluence_x_confidence'] = (confluence_score / 4.0) * (trade.get('confidence', 50) / 100.0)
    features['sweep_x_fvg'] = trade.get('sweep', 0) * trade.get('fvg', 0)
    features['volatility_x_risk'] = features['atr_expansion'] * features['risk_pct']

    return features


def calculate_historical_context(history):
    """Calculate context metrics from trade history"""
    
    if len(history) < 5:
        return {
            'recent_win_rate': 0.5,
            'streak_count': 0,
            'cumulative_pnl': 0,
            'current_dd_pct': 0
        }
    
    recent_trades = history[-10:]
    recent_wins = sum(1 for t in recent_trades if t.get('status') == 'WIN')
    recent_win_rate = recent_wins / len(recent_trades) if recent_trades else 0.5
    
    streak = 0
    streak_sign = None
    for trade in reversed(history):
        is_win = trade.get('status') == 'WIN'
        if streak_sign is None:
            streak_sign = is_win
            streak = 1
        elif (is_win and streak_sign) or (not is_win and not streak_sign):
            streak += 1
        else:
            break
    
    if streak_sign is False:
        streak = -streak
    
    cumulative_pnl = sum(t.get('pnl', 0) for t in history[-20:])
    
    running_pnl = []
    cumsum = 0
    for trade in history[-20:]:
        cumsum += trade.get('pnl', 0)
        running_pnl.append(cumsum)
    
    peak = max(running_pnl) if running_pnl else 0
    current = running_pnl[-1] if running_pnl else 0
    drawdown_pct = ((peak - current) / max(abs(peak), 0.01)) * 100 if peak > 0 else 0
    
    return {
        'recent_win_rate': recent_win_rate,
        'streak_count': streak,
        'cumulative_pnl': cumulative_pnl,
        'current_dd_pct': drawdown_pct
    }


def analyze_exit(trade):
    """Analyze WHY a trade exited (SL vs TP)"""
    
    analysis = {
        'trade_no': trade.get('trade_no'),
        'symbol': trade.get('symbol'),
        'direction': trade.get('direction'),
        'entry': trade.get('entry'),
        'exit_price': trade.get('exit_price'),
        'status': trade.get('status'),
        'pnl': trade.get('pnl', 0),
        'timestamp': datetime.now().isoformat()
    }
    
    if trade.get('status') == 'WIN':
        analysis['reason'] = 'TOOK_PROFIT'
        analysis['reason_detail'] = 'Trade reached take profit target'
        analysis['hit_adversity'] = False
    else:
        analysis['reason'] = 'STOP_LOSS'
        analysis['reason_detail'] = 'Trade hit stop loss'
        analysis['hit_adversity'] = True
    
    return analysis


def train_model_incremental():
    """Retrain model after EACH trade closes"""
    
    history = get_trade_history()
    
    if len(history) < 5:
        logger.info(f"⏳ Waiting for trades to learn... ({len(history)}/5)")
        return None

    logger.info(f"\n🧠 CONTINUOUS LEARNING: Training on {len(history)} trades...")

    context = calculate_historical_context(history)

    data = []
    for i, trade in enumerate(history):
        if trade.get("status") in ["WIN", "LOSS"]:
            feat = extract_pro_features_from_trade(trade, context)
            feat['target'] = 1 if trade["status"] == "WIN" else 0
            
            exit_analysis = analyze_exit(trade)
            feat['exit_reason'] = exit_analysis['reason']
            feat['hit_adversity'] = exit_analysis['hit_adversity']
            
            data.append(feat)

    if len(data) < 5:
        return None

    df = pd.DataFrame(data)
    X = df.drop('target', axis=1, errors='ignore')
    y = df['target']

    # === FIX: Convert to pure numeric features (removes strings like 'exit_reason') ===
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.fillna(0)  # Fill any NaNs

    logger.info(f"Training with {X.shape[1]} numeric features")

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.5,
        reg_alpha=0.3,
        reg_lambda=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    model.fit(X, y, verbose=0)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(X.columns.tolist(), "models/feature_names.pkl")
    
    accuracy = model.score(X, y)
    win_count = (y == 1).sum()
    loss_count = (y == 0).sum()
    win_rate = (y == 1).sum() / len(y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info(f"\n✅ MODEL UPDATED!")
    logger.info(f"   Trades Learned: {len(data)} (W: {win_count}, L: {loss_count})")
    logger.info(f"   Accuracy: {accuracy:.1%}")
    logger.info(f"   Win Rate: {win_rate:.1%}")
    
    logger.info(f"\n   📊 Top 10 Features (What Model Learned):")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"      {row['feature']}: {row['importance']:.3f}")

    logger.info(f"\n   📈 Latest Trade Analysis:")
    if len(history) > 0:
        last_trade = history[-1]
        exit_analysis = analyze_exit(last_trade)
        
        logger.info(f"      Trade #{exit_analysis['trade_no']}: {exit_analysis['symbol']}")
        logger.info(f"      Direction: {exit_analysis['direction']}")
        logger.info(f"      Entry → Exit: ${exit_analysis['entry']:.6f} → ${exit_analysis['exit_price']:.6f}")
        logger.info(f"      Result: {exit_analysis['status']} ({exit_analysis['pnl']:+.2f}%)")
        logger.info(f"      Exit Reason: {exit_analysis['reason_detail']}")

    return model


def get_xgboost_probability(trade_features):
    """Get AI probability for a trade setup"""
    
    if not Path(MODEL_PATH).exists():
        return 50.0

    try:
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load("models/feature_names.pkl")
        
        X = pd.DataFrame([trade_features])
        X = X[feature_names]
        
        prob = model.predict_proba(X)[0][1] * 100
        
        logger.debug(f"AI Probability: {prob:.1f}%")
        return round(prob, 1)
        
    except Exception as e:
        logger.error(f"XGBoost prediction failed: {e}")
        return 50.0
