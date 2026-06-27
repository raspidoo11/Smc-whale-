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
        "daily_pnl": 0.0,
        "peak_daily_pnl": 0.0
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


def risk_amount():
    data = get_balance()
    return data.get("balance", 100.0) * 0.01


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
            trade["exit_price"] = exit_price
            closed_trade = trade
        else:
            remaining.append(trade)

    if closed_trade:
        history.append(closed_trade)
        save_trade_history(history)

    save_open_trades(remaining)
    return closed_trade


def get_trade_history():
    return load_json(HISTORY_FILE, [])


def save_trade_history(history):
    save_json(HISTORY_FILE, history)


def update_balance(pnl):
    data = get_balance()
    data["balance"] += pnl
    data["daily_pnl"] += pnl
    if data["daily_pnl"] > data["peak_daily_pnl"]:
        data["peak_daily_pnl"] = data["daily_pnl"]
    save_balance(data)
    logger.info(f"Balance updated: ${data['balance']:.2f}")
    return data


def trading_allowed():
    data = get_balance()
    if data["peak_daily_pnl"] >= 10 and data["daily_pnl"] <= 10:
        logger.info("Daily target protected.")
        return False
    return True
