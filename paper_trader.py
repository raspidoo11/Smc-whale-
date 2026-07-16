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

    # Status from realized PnL sign (not exit_reason string matching).
    # Trailing-stop exits are floored at fee-aware breakeven in the monitor,
    # so an armed trail should never land here as a LOSS. Still: if the exit
    # is on the favorable side of entry, treat tiny fee-rounding scratches
    # on trail hits as WIN so cooldown/stats don't punish a worked setup.
    if pnl_after_fees > 0:
        status = "WIN"
    elif (
        exit_reason == "Trailing Stop Hit"
        and (
            (direction == "LONG" and exit_price >= entry)
            or (direction == "SHORT" and exit_price <= entry)
        )
    ):
        status = "WIN"
    else:
        status = "LOSS"

    # --- FIX #2: pnl/fees used to be set on the dict AFTER close_trade()
    # had already saved it to trade_history.json, so they never actually
    # persisted (pnl stayed None forever). Passing them as extra_fields
    # means they're merged onto the trade before it's written.
    closed_trade = close_trade(
        trade["symbol"],
        exit_price,
        status,
        extra_fields={
            "pnl": round(pnl_after_fees, 2),
            "entry_fee": trade.get("entry_fee", 0),
            "exit_fee": exit_fee,
            "exit_reason": exit_reason,
        },
    )

    # --- FIX #3: only touch the balance if this call actually closed the
    # trade. close_trade() returns None when the trade was already closed by
    # another path (e.g. reconcile got there first) — updating the balance
    # anyway double-counted the PnL and fired a second close alert.
    if closed_trade is None:
        logger.warning(
            f"↩️ {trade['symbol']}: already closed elsewhere — skipping balance/alert"
        )
        return None

    update_balance(pnl_after_fees)

    logger.info(
        f"✅ CLOSED {trade['symbol']} | {exit_reason} | {status} | "
        f"Raw PnL: ${pnl:.2f} | Exit Fee: ${exit_fee:.2f} | "
        f"Net PnL: ${pnl_after_fees:.2f}"
    )

    return pnl_after_fees
