import logging
from exchange import get_exchange
from trade_manager import get_open_trades, close_trade, update_balance, get_balance, save_open_trades
from telegram_alerts import send_alert

logger = logging.getLogger(__name__)
exchange = get_exchange()


async def get_current_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return float(ticker.get("last", 0))
    except:
        return None


async def monitor_trades():
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

        if hit_tp or hit_sl:
            result = "WIN" if hit_tp else "LOSS"
            close_trade(symbol, current_price, result)
            update_balance(pnl)
            bal = get_balance()["balance"]

            emoji = "✅" if hit_tp else "❌"
            await send_alert(
                f"""
{emoji} PAPER {result}

📊 {symbol}

━━━━━━━━━━━━━━

📌 Entry: {entry:.6f}

🏁 Exit: {current_price:.6f}

{'💵 Profit' if hit_tp else '💸 Loss'}: {'+' if hit_tp else '-'}${abs(pnl):.2f}

💰 Balance: ${bal:.2f}

━━━━━━━━━━━━━━

{'🎉 TP HIT' if hit_tp else '🛑 STOP LOSS HIT'}
"""
            )
            continue

        # Break-Even and Trailing Stop (only if not hit)

    save_open_trades(trades)
