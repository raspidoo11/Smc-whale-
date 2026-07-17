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
        "entry_fee_applied": True,  # prepaid — close must not charge again
    }

    add_trade(trade)
    logger.info(
        f"✅ OPENED: {signal['direction']} {signal.get('symbol')} "
        f"| Qty: {qty} | Max Risk: ${MAX_RISK_PER_TRADE_USD} | Entry Fee: ${entry_fee:.2f}"
    )
    return trade


def close_paper_trade_with_fees(trade: dict, exit_price: float, exit_reason: str):
    """Close a paper (or local-mirror) trade and credit paper balance.

    Balance is updated *inside* close_trade via balance_delta so history and
    cash cannot diverge (the trailing-stop bug: WIN rows with a frozen
    balance happened when close succeeded but a separate update_balance
    never ran, or ran against a trade already removed from the open list).
    """
    entry = float(trade["entry"])
    qty = float(trade["qty"])
    direction = str(trade.get("direction", "LONG")).upper()
    exit_price = float(exit_price)

    if qty <= 0:
        logger.error(f"❌ {trade.get('symbol')}: qty={qty} — cannot close with zero size")
        return None

    if direction == "LONG":
        pnl = (exit_price - entry) * qty
    else:
        pnl = (entry - exit_price) * qty

    exit_fee = calculate_exit_fee(exit_price, qty)

    # Entry fee: open_paper_trade deducts it up front; main.py's paper path
    # does not. Charge it on close when it was never applied so trail/SL/TP
    # wins still move the paper balance by the true net, not gross-minus-exit.
    entry_fee = float(trade.get("entry_fee") or 0)
    entry_fee_prepaid = bool(trade.get("entry_fee_applied"))
    if entry_fee <= 0 and not entry_fee_prepaid:
        entry_fee = calculate_entry_fee(entry, qty)
    if entry_fee_prepaid:
        # Already removed from balance at open — only subtract exit fee here.
        pnl_after_fees = pnl - exit_fee
    else:
        pnl_after_fees = pnl - exit_fee - entry_fee

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

    # Persist pnl/fees AND apply balance_delta in one close_trade call.
    closed_trade = close_trade(
        trade["symbol"],
        exit_price,
        status,
        extra_fields={
            "pnl": round(pnl_after_fees, 4),
            "entry_fee": entry_fee,
            "exit_fee": exit_fee,
            "exit_reason": exit_reason,
        },
        balance_delta=pnl_after_fees,
        trade_no=trade.get("trade_no"),
    )

    # close_trade() returns None when the trade was already closed by another
    # path (e.g. reconcile) — balance was not touched again (no double count).
    if closed_trade is None:
        logger.warning(
            f"↩️ {trade['symbol']}: already closed elsewhere — skipping balance/alert"
        )
        return None

    logger.info(
        f"✅ CLOSED {trade['symbol']} | {exit_reason} | {status} | "
        f"Raw PnL: ${pnl:.4f} | Fees: ${entry_fee + exit_fee:.4f} | "
        f"Net PnL: ${pnl_after_fees:.4f}"
    )

    return pnl_after_fees
