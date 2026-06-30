import json
import os
import logging

logger = logging.getLogger(__name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

BALANCE_FILE = os.path.join(DATA_DIR, "paper_balance.json")
OPEN_TRADES_FILE = os.path.join(DATA_DIR, "open_trades.json")
HISTORY_FILE = os.path.join(DATA_DIR, "trade_history.json")

def load_json(file_path, default):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def get_balance():
    default = {
        "balance": 100.0,
        "daily_pnl": 0.0
    }
    data = load_json(BALANCE_FILE, default)
    
    for key, value in default.items():
        if key not in data:
            data[key] = value
    
    return data

def save_balance(data):
    save_json(BALANCE_FILE, data)

def get_open_trades():
    return load_json(OPEN_TRADES_FILE, [])

def save_open_trades(trades):
    save_json(OPEN_TRADES_FILE, trades)

def get_trade_history():
    return load_json(HISTORY_FILE, [])

def save_trade_history(history):
    save_json(HISTORY_FILE, history)

def risk_amount():
    data = get_balance()
    return data.get("balance", 100.0) * 0.01

def trade_exists(symbol):
    trades = get_open_trades()
    
    for trade in trades:
        if (
            trade["symbol"] == symbol
            and trade["status"] == "OPEN"
        ):
            return True
    
    return False

def next_trade_number():
    history = get_trade_history()
    open_trades = get_open_trades()
    
    return len(history) + len(open_trades) + 1

def add_trade(trade):
    trades = get_open_trades()
    trades.append(trade)
    save_open_trades(trades)
    logger.info(f"Trade added: {trade['symbol']}")

def close_trade(symbol, exit_price, result):
    trades = get_open_trades()
    history = get_trade_history()

    remaining = []
    closed_trade = None

    for trade in trades:
        if (
            trade.get("symbol") == symbol
            and trade.get("status") == "OPEN"
            and closed_trade is None
        ):
            trade["status"] = result
            trade["exit_price"] = float(exit_price)
            closed_trade = trade
        else:
            remaining.append(trade)

    if closed_trade:
        history.append(closed_trade)
        save_trade_history(history)
        logger.info(f"Trade closed: {symbol} ({result})")

    save_open_trades(remaining)

    return closed_trade

def update_balance(pnl):
    data = get_balance()
    data["balance"] += pnl
    data["daily_pnl"] += pnl
    
    save_balance(data)
    logger.info(f"Balance updated: ${data['balance']:.2f} | Daily PnL: ${data['daily_pnl']:.2f}")
    
    return data

def trading_allowed():
    """
    FIXED: Only block trading if LOSSES hit -$15 (15% drawdown from $100)
    NO profit cap - let wins run!
    """
    
    # Allow override
    if os.getenv("DAILY_PROTECTION", "true").lower() == "false":
        return True
    
    data = get_balance()
    daily_pnl = data.get("daily_pnl", 0)
    
    # ONLY STOP IF LOSING $15 OR MORE
    if daily_pnl <= -15:
        logger.warning(f"⛔ LOSS LIMIT HIT: Daily PnL ${daily_pnl:.2f} <= -$15")
        logger.warning("🛑 Trading paused for rest of day (loss protection)")
        return False
    
    # Otherwise, keep trading! Let profits run!
    return True

def reset_daily_pnl():
    """
    Call this at start of each trading day (midnight)
    """
    data = get_balance()
    data["daily_pnl"] = 0.0
    save_balance(data)
    logger.info("🔄 Daily PnL reset for new trading day")
