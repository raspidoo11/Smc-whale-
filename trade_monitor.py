import logging
from trade_manager import get_open_trades, save_open_trades, get_current_price, close_trade, update_balance, get_balance
from paper_trader import apply_fees
from telegram_alerts import send_alert

logger = logging.getLogger(__name__)


def monitor_trades():
    open_trades = get_open_trades()
    if not open_trades:
        return

    updated_trades = []
    modified = False

    for trade in open_trades:
        symbol = trade["symbol"]
        current_price = get_current_price(symbol)

        if current_price is None:
            updated_trades.append(trade)
            continue

        entry = float(trade["entry"])
        sl = float(trade["sl"])
        tp = float(trade["tp"])
        direction = trade["direction"]
        qty = float(trade["qty"])

        hit_tp = False
        hit_sl = False

        if direction == "LONG":
            hit_tp = current_price >= tp
            hit_sl = current_price <= sl
        else:  # SHORT
            hit_tp = current_price <= tp
            hit_sl = current_price >= sl

        if hit_tp or hit_sl:
            exit_price = tp if hit_tp else sl
            exit_reason = "Take Profit Hit" if hit_tp else "Stop Loss Hit"

            # Calculate raw PnL
            if direction == "LONG":
                pnl = (exit_price - entry) * qty
            else:
                pnl = (entry - exit_price) * qty

            # Apply realistic fees
            pnl_after_fees = apply_fees(pnl, entry, qty)

            # Update balance
            update_balance(pnl_after_fees)

            # Close trade using existing function
            close_trade(symbol, exit_price, "WIN" if hit_tp else "LOSS")

            # Send alert with fee-applied PnL
            send_alert(
                f"{'✅' if hit_tp else '❌'} {exit_reason}\n"
                f"{direction} {symbol}\n"
                f"Entry → Exit: ${entry:.4f} → ${exit_price:.4f}\n"
                f"PnL: ${pnl_after_fees:.2f} (after fees)\n"
                f"New Balance: ${get_balance():.2f}"
            )

            modified = True
        else:
            updated_trades.append(trade)

    if modified:
        save_open_trades(updated_trades)
