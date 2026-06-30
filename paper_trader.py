import logging
from trade_manager import get_balance, update_balance, add_trade, close_trade

logger = logging.getLogger(__name__)

MAX_RISK_PER_TRADE_USD = 5.0
FEE_RATE = 0.0004
LEVERAGE = 10


def get_numeric_balance():
    """Safely extract numeric balance (handles if get_balance returns dict or float)"""
    bal = get_balance()
    if isinstance(bal, dict):
        return float(bal.get("USDT", bal.get("balance", bal.get("total", 0))))
    return float(bal)


def calculate_qty(entry: float, sl: float) -> float:
    """Calculate quantity so max loss = $5"""
    balance = get_numeric_balance()
    if entry == sl or entry == 0:
        return 0.0

    risk_per_unit = abs(entry - sl)
    entry_fee_rate = FEE_RATE
    risk_adjusted = MAX_RISK_PER_TRADE_USD / (1 + entry_fee_rate)
    raw_qty = risk_adjusted / risk_per_unit
    max_position_value = balance * LEVERAGE
    max_qty_by_leverage = max_position_value / entry
    qty = min(raw_qty, max_qty_by_leverage)
    return round(qty, 6)


def calculate_entry_fee(entry_price: float, quantity: float) -> float:
    entry_value = entry_price * quantity
    fee = entry_value * FEE_RATE
    return round(fee, 2)


def calculate_exit_fee(exit_price: float, quantity: float) -> float:
    exit_value = exit_price * quantity
    fee = exit_value * FEE_RATE
    return round(fee, 2)


def open_paper_trade(signal: dict):
    balance = get_numeric_balance()
    entry = signal["entry"]
    sl = signal["sl"]
    qty = calculate_qty(entry, sl)

    if qty <= 0:
        logger.warning("❌ Quantity too small or invalid SL. Trade not opened.")
        return None

    entry_fee = calculate_entry_fee(entry, qty)
    new_balance = balance - entry_fee

    if new_balance < 0:
        logger.error(f"❌ Insufficient balance. Need ${entry_fee}, have ${balance}")
        return None

    update_balance(-entry_fee)

    trade = {
        "symbol": signal.get("symbol"),
        "direction": signal["direction"],
        "entry": float(entry),
        "sl": float(sl),
        "tp": float(signal["tp"]),
        "qty": float(qty),
        "leverage": LEVERAGE,
        "status": "OPEN",
        "confidence": signal.get("confidence"),
        "ai_prob": signal.get("ai_prob"),
        "entry_fee": entry_fee,
    }

    add_trade(trade)
    logger.info(
        f"✅ OPENED: {signal['direction']} {signal.get('symbol')} "
        f"| Qty: {qty} | Max Risk: ${MAX_RISK_PER_TRADE_USD} | Entry Fee: ${entry_fee:.2f}"
    )
    return trade


def close_paper_trade_with_fees(trade: dict, exit_price: float, exit_reason: str):
    entry = float(trade["entry"])
    qty = float(trade["qty"])
    direction = trade["direction"]

    if direction == "LONG":
        pnl = (exit_price - entry) * qty
    else:
        pnl = (entry - exit_price) * qty

    exit_fee = calculate_exit_fee(exit_price, qty)
    pnl_after_fees = pnl - exit_fee

    update_balance(pnl_after_fees)
    close_trade(trade["symbol"], exit_price, "WIN" if exit_reason == "Take Profit Hit" else "LOSS")

    logger.info(
        f"✅ CLOSED {trade['symbol']} | {exit_reason} | "
        f"Raw PnL: ${pnl:.2f} | Exit Fee: ${exit_fee:.2f} | Net PnL: ${pnl_after_fees:.2f}"
    )
    return pnl_after_fees
