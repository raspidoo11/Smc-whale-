import logging
from exchange import get_exchange
from trade_manager import (
    get_open_trades, close_trade, update_balance, get_balance
)
from telegram_alerts import send_alert

logger = logging.getLogger(__name__)
exchange = get_exchange()


async def get_current_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return float(ticker.get("last", 0))
    except Exception as e:
        logger.error(f"Price fetch failed for {symbol}: {e}")
        return None


async def monitor_trades():
    """Check open paper trades for TP/SL hits"""
    trades = get_open_trades()
    if not trades:
        return

    logger.info(f"Monitoring {len(trades)} open paper trades...")

    for trade in trades[:]:
        if trade.get("status") != "OPEN":
            continue

        symbol = trade["symbol"]
        entry = float(trade["entry"])
        sl = float(trade["sl"])
        tp = float(trade["tp"])
        direction = trade["direction"]
        qty = float(trade.get("qty", 1))

        current_price = await get_current_price(symbol)
        if not current_price:
            continue

        if direction == "LONG":
            pnl = (current_price - entry) * qty
            hit_tp = current_price >= tp
            hit_sl = current_price <= sl
        else:
            pnl = (entry - current_price) * qty
            hit_tp = current_price <= tp
            hit_sl = current_price >= sl

        if hit_tp:
            close_trade(symbol, current_price, "WIN")
            update_balance(pnl)
            bal = get_balance()["balance"]
            await send_alert(
                f"✅ PAPER WIN\n\n"
                f"{symbol}\n"
                f"Entry: {entry:.4f}\n"
                f"Exit: {current_price:.4f}\n"
                f"Profit: +${pnl:.2f}\n"
                f"Balance: ${bal:.2f}"
            )
            continue

        if hit_sl:
            close_trade(symbol, current_price, "LOSS")
            update_balance(pnl)
            bal = get_balance()["balance"]
            await send_alert(
                f"❌ PAPER LOSS\n\n"
                f"{symbol}\n"
                f"Entry: {entry:.4f}\n"
                f"Exit: {current_price:.4f}\n"
                f"Loss: -${abs(pnl):.2f}\n"
                f"Balance: ${bal:.2f}"
            )
            continue

        # TODO: Add Break-Even and Trailing Stop later
