import logging
from trade_manager import get_open_trades, save_open_trades, get_current_price
from paper_trader import close_paper_trade
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

        entry = trade["entry"]
        sl = trade["sl"]
        tp = trade["tp"]
        direction = trade["direction"]

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

            # Use the new centralized close function (applies fees automatically)
            close_paper_trade(
                trade_id=trade.get("id"),
                exit_price=exit_price,
                exit_reason=exit_reason
            )

            # Send alert
            send_alert(
                f"{'✅' if hit_tp else '❌'} {exit_reason}\n"
                f"{direction} {symbol}\n"
                f"Entry: ${entry}\n"
                f"Exit: ${exit_price}\n"
                f"New Balance: ${get_balance():.2f}"
            )

            modified = True
        else:
            updated_trades.append(trade)

    if modified:
        save_open_trades(updated_trades)
