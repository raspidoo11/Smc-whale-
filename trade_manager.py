import json
import os
import logging
from datetime import datetime, timedelta, timezone

from config import DATA_DIR, DAILY_LOSS_LIMIT

logger = logging.getLogger(__name__)

BALANCE_FILE = os.path.join(DATA_DIR, "paper_balance.json")
OPEN_TRADES_FILE = os.path.join(DATA_DIR, "open_trades.json")
HISTORY_FILE = os.path.join(DATA_DIR, "trade_history.json")
SIGNAL_HASH_FILE = os.path.join(DATA_DIR, "signal_hashes.json")
COOLDOWN_FILE = os.path.join(DATA_DIR, "cooldowns.json")

# Storage backend: "sqlite" (default) or "json". SQLite gives atomicity + WAL
# concurrency + queryable history and auto-migrates existing JSON on first run.
# The JSON backend is kept as a fallback / for tests.
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "sqlite").lower()

# ==================== JSON HELPERS ====================

def load_json(file_path, default):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(file_path, data):
    """Atomic write: serialize to a temp file, fsync, then os.replace() into
    place. Prevents a killed/restarted process (Railway redeploys, OOM, etc.)
    from leaving a half-written, corrupt JSON file that would wipe balances or
    open trades on next load. default=str keeps numpy/datetime values safe."""
    tmp_path = f"{file_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=4, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, file_path)

# ==================== CORE FUNCTIONS ====================
# Primitive get/save dispatch to the selected backend. Everything below these
# (add_trade, close_trade, next_trade_number, balance/hash/cooldown logic) is
# backend-agnostic — it only calls these primitives — so the SQLite switch
# touches nothing else.

_BALANCE_DEFAULT = {"balance": 100.0, "daily_pnl": 0.0}


def get_balance():
    if STORAGE_BACKEND == "sqlite":
        import db
        data = db.kv_get("balance", dict(_BALANCE_DEFAULT))
    else:
        data = load_json(BALANCE_FILE, dict(_BALANCE_DEFAULT))
    for key, value in _BALANCE_DEFAULT.items():
        if key not in data:
            data[key] = value
    return data

def save_balance(data):
    if STORAGE_BACKEND == "sqlite":
        import db
        db.kv_set("balance", data)
    else:
        save_json(BALANCE_FILE, data)

def get_open_trades():
    if STORAGE_BACKEND == "sqlite":
        import db
        return db.read_list("open_trades")
    return load_json(OPEN_TRADES_FILE, [])

def save_open_trades(trades):
    if STORAGE_BACKEND == "sqlite":
        import db
        db.write_list("open_trades", trades)
    else:
        save_json(OPEN_TRADES_FILE, trades)

def get_trade_history():
    if STORAGE_BACKEND == "sqlite":
        import db
        return db.read_list("history")
    return load_json(HISTORY_FILE, [])

def save_trade_history(history):
    if STORAGE_BACKEND == "sqlite":
        import db
        db.write_list("history", history)
    else:
        save_json(HISTORY_FILE, history)

def next_trade_number():
    # Backtest-backfilled rows are training data, not trades this bot took —
    # they must not inflate the visible trade numbering in alerts.
    real_history = [t for t in get_trade_history() if t.get("source") != "backtest"]
    return len(real_history) + len(get_open_trades()) + 1

def add_trade(trade):
    trades = get_open_trades()
    trades.append(trade)
    save_open_trades(trades)
    logger.info(f"Trade added: {trade['symbol']}")

def close_trade(symbol, exit_price, result, extra_fields=None):
    """
    extra_fields: optional dict merged into the trade BEFORE it's written to
    trade_history.json (e.g. {"pnl": ..., "entry_fee": ..., "exit_fee": ...}).

    Previously, callers (paper_trader.py) mutated the dict this function
    *returned*, AFTER save_trade_history() had already run. Since
    get_trade_history() always does a fresh load_json() from disk (there's
    no in-memory cache), those late mutations never made it back to disk --
    pnl was permanently None in trade_history.json even though it was
    correctly calculated in memory. Passing extra_fields in means pnl/fees
    are on the dict before it's ever saved.
    """
    trades = get_open_trades()
    history = get_trade_history()
    remaining = []
    closed_trade = None

    for trade in trades:
        if trade.get("symbol") == symbol and trade.get("status") == "OPEN" and closed_trade is None:
            trade["status"] = result
            trade["exit_price"] = float(exit_price)
            if extra_fields:
                trade.update(extra_fields)
            closed_trade = trade
        else:
            remaining.append(trade)

    if closed_trade:
        history.append(closed_trade)
        save_trade_history(history)
        logger.info(f"Trade closed: {symbol} ({result})")

        # NOTE: this used to check `result == "SL"`, but result is always
        # "WIN" or "LOSS" (see paper_trader.py's close_paper_trade_with_fees),
        # so that cooldown never actually fired. Fixed to match real values.
        if result == "LOSS":
            set_cooldown(symbol, minutes=60)

    save_open_trades(remaining)
    return closed_trade

def update_balance(pnl):
    data = get_balance()
    data["balance"] += pnl
    data["daily_pnl"] += pnl
    save_balance(data)
    return data

def trading_allowed():
    if os.getenv("DAILY_PROTECTION", "true").lower() == "false":
        return True
    return get_balance().get("daily_pnl", 0) > -DAILY_LOSS_LIMIT

def reset_daily_pnl():
    data = get_balance()
    data["daily_pnl"] = 0.0
    save_balance(data)
    logger.info("🔄 Daily PnL reset")

# ==================== RISK MANAGEMENT ====================

def risk_amount():
    """Legacy function - kept for backward compatibility"""
    return get_balance().get("balance", 100.0) * 0.01

def get_risk_amount(leverage: int = 10) -> float:
    """
    Dynamic risk amount.
    Default: 0.5% of account balance (configurable via RISK_PERCENT env var)
    """
    balance = get_balance().get("balance", 100.0)
    risk_percent = float(os.getenv("RISK_PERCENT", "0.5")) / 100
    return round(balance * risk_percent, 2)

def get_risk_percent() -> float:
    return float(os.getenv("RISK_PERCENT", "0.5"))

# ==================== TRADE EXISTENCE CHECK ====================

def trade_exists(symbol):
    trades = get_open_trades()
    return any(
        t.get("symbol") == symbol and t.get("status") == "OPEN"
        for t in trades
    )

# ==================== SIGNAL HASH SYSTEM ====================

def get_signal_hashes():
    if STORAGE_BACKEND == "sqlite":
        import db
        return db.kv_get("signal_hashes", [])
    return load_json(SIGNAL_HASH_FILE, [])

def save_signal_hashes(hashes):
    if STORAGE_BACKEND == "sqlite":
        import db
        db.kv_set("signal_hashes", hashes)
    else:
        save_json(SIGNAL_HASH_FILE, hashes)

def get_signal_hash_exists(signal_hash):
    if not signal_hash:
        return False
    return signal_hash in get_signal_hashes()

def save_signal_hash(signal_hash):
    if not signal_hash:
        return
    hashes = get_signal_hashes()
    if signal_hash not in hashes:
        hashes.append(signal_hash)
        if len(hashes) > 500:
            hashes = hashes[-500:]
        save_signal_hashes(hashes)

# ==================== COOLDOWN SYSTEM ====================

def get_cooldowns():
    if STORAGE_BACKEND == "sqlite":
        import db
        return db.kv_get("cooldowns", {})
    return load_json(COOLDOWN_FILE, {})

def save_cooldowns(cooldowns):
    if STORAGE_BACKEND == "sqlite":
        import db
        db.kv_set("cooldowns", cooldowns)
    else:
        save_json(COOLDOWN_FILE, cooldowns)

def is_symbol_in_cooldown(symbol):
    cooldowns = get_cooldowns()
    if symbol not in cooldowns:
        return False
    cooldown_until = datetime.fromisoformat(cooldowns[symbol])
    return datetime.now(timezone.utc) < cooldown_until

def set_cooldown(symbol, minutes=60):
    cooldowns = get_cooldowns()
    cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    cooldowns[symbol] = cooldown_until.isoformat()
    save_cooldowns(cooldowns)
    logger.info(f"⏳ Cooldown set for {symbol} until {cooldown_until.strftime('%H:%M')} UTC")

def get_cooldown_remaining(symbol):
    cooldowns = get_cooldowns()
    if symbol not in cooldowns:
        return 0
    cooldown_until = datetime.fromisoformat(cooldowns[symbol])
    remaining = (cooldown_until - datetime.now(timezone.utc)).total_seconds()
    return max(0, int(remaining / 60))
