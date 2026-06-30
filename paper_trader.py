import logging
from trade_manager import get_balance, update_balance, add_trade, close_trade

logger = logging.getLogger(__name__)

# ==================== CONFIG ====================
MAX_RISK_PER_TRADE_USD = 5.0          # Fixed $5 max loss per trade (forever)
FEE_RATE = 0.0004                     # 0.04% round-trip fee (realistic for Bybit futures)
LEVERAGE = 10
# ===============================================


def calculate_qty(entry: float, sl: float, balance: float) -> float:
    """
    Calculate quantity so that the maximum possible loss = $MAX_RISK_PER_TRADE_USD ($5)
    """
    if entry == sl or entry == 0:
        return 0.0

    risk_per_unit = abs(entry - sl)
    
    # Quantity so risk = exactly $5
    raw_qty = MAX_RISK_PER_TRADE_USD / risk_per_unit
    
    # Respect leverage limit
    max_position_value = balance * LEVERAGE
    max_qty_by_leverage = max_position_value / entry
    
    qty = min(raw_qty, max_qty_by_leverage)
    
    return round(qty, 6)


def apply_fees(pnl: float, entry_price: float, quantity: float) -> float:
    """
    Apply realistic round-trip fees to the PnL
    """
    entry_value = entry_price * quantity
    fees = entry_value * FEE_RATE
    return pnl - fees


def open_paper_trade(signal: dict):
    """
    Open a paper trade with proper $5 risk sizing + fees
    """
    balance = get_balance()
    
    entry = signal["entry"]
    sl = signal["sl"]
    
    qty = calculate_qty(entry, sl, balance)
    
    if qty <= 0:
        logger.warning("Quantity too small or invalid SL. Trade not opened.")
        return None
    
    # Record the trade
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
    logger.info(f"Opened paper trade: {signal['direction']} {signal.get('symbol')} | Qty: {qty} | Risk: ${MAX_RISK_PER_TRADE_USD}")
    
    return trade


def close_paper_trade(trade_id: int, exit_price: float, exit_reason: str):
    """
    Close a paper trade, calculate PnL, apply fees, and update balance.
    Call this from trade_monitor.py when SL or TP is hit.
    """
    trade = get_trade_by_id(trade_id)
    if not trade:
        logger.error(f"Trade #{trade_id} not found")
        return None

    entry = float(trade["entry"])
    qty = float(trade["qty"])
    direction = trade["direction"]

    # Calculate raw PnL
    if direction == "LONG":
        pnl = (exit_price - entry) * qty
    else:
        pnl = (entry - exit_price) * qty

    # Apply realistic fees
    pnl_after_fees = apply_fees(pnl, entry, qty)

    # Update account balance
    update_balance(pnl_after_fees)

    # Update trade record
    update_trade(trade_id, {
        "status": "CLOSED",
        "exit_price": exit_price,
        "pnl": round(pnl_after_fees, 2),
        "exit_reason": exit_reason
    })

    logger.info(f"Closed #{trade_id} | {exit_reason} | PnL: ${pnl_after_fees:.2f} (after fees)")
    return pnl_after_fees
