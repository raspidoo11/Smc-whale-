import logging
from trade_manager import get_balance, update_balance, add_trade, close_trade

logger = logging.getLogger(__name__)

MAX_RISK_PER_TRADE_USD = 5.0
FEE_RATE = 0.0004
LEVERAGE = 10


def calculate_qty(entry: float, sl: float, balance: float) -> float:
    """Calculate quantity so that hitting SL loses exactly $5"""
    if entry == sl or entry == 0:
        return 0.0

    risk_per_unit = abs(entry - sl)
    raw_qty = MAX_RISK_PER_TRADE_USD / risk_per_unit
    max_position_value = balance * LEVERAGE
    max_qty_by_leverage = max_position_value / entry
    qty = min(raw_qty, max_qty_by_leverage)
    return round(qty, 6)


def apply_fees(pnl: float, entry_price: float, quantity: float) -> float:
    """Apply realistic round-trip fees"""
    entry_value = entry_price * quantity
    fees = entry_value * FEE_RATE
    return pnl - fees


def open_paper_trade(signal: dict):
    """Open a paper trade with $5 max risk"""
    balance = get_balance()
    entry = signal["entry"]
    sl = signal["sl"]
    qty = calculate_qty(entry, sl, balance)

    if qty <= 0:
        logger.warning("Quantity too small or invalid SL. Trade not opened.")
        return None

    trade = {
        "symbol": signal.get("symbol"),
        "direction": signal["direction"],
        "entry": entry,
        "sl": sl,
        "tp": signal["tp"],
        "qty": qty,
        "leverage": LEVERAGE,
        "status": "OPEN",
        "confidence": signal.get("confidence"),
        "ai_prob": signal.get("ai_prob"),
    }

    add_trade(trade)
    logger.info(f"Opened: {signal['direction']} {signal.get('symbol')} | Qty: {qty} | Max Risk: ${MAX_RISK_PER_TRADE_USD}")
    return trade


def close_paper_trade_with_fees(trade: dict, exit_price: float, exit_reason: str):
    """
    Close trade, apply fees, update balance.
    Use this in trade_monitor.py when SL or TP is hit.
    """
    entry = float(trade["entry"])
    qty = float(trade["qty"])
    direction = trade["direction"]

    # Calculate raw PnL
    if direction == "LONG":
        pnl = (exit_price - entry) * qty
    else:
        pnl = (entry - exit_price) * qty

    # Apply fees
    pnl_after_fees = apply_fees(pnl, entry, qty)

    # Update balance
    update_balance(pnl_after_fees)

    # Close trade using existing function
    close_trade(trade["symbol"], exit_price, "WIN" if exit_reason == "Take Profit Hit" else "LOSS")

    logger.info(f"Closed {trade['symbol']} | {exit_reason} | PnL after fees: ${pnl_after_fees:.2f}")
    return pnl_after_fees
