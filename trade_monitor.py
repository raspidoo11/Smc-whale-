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

            if hit_tp:
                await send_alert(
                    f"""
✅ PAPER WIN

📊 {symbol}

━━━━━━━━━━━━━━

📌 Entry: {entry:.6f}

🏁 Exit: {current_price:.6f}

💵 Profit: +${pnl:.2f}

💰 Balance: ${bal:.2f}

━━━━━━━━━━━━━━

🎉 TP HIT
"""
                )
            else:
                await send_alert(
                    f"""
❌ PAPER LOSS

📊 {symbol}

━━━━━━━━━━━━━━

📌 Entry: {entry:.6f}

🏁 Exit: {current_price:.6f}

💸 Loss: -${abs(pnl):.2f}

💰 Balance: ${bal:.2f}

━━━━━━━━━━━━━━

🛑 STOP LOSS HIT
"""
                )
            continue

        # Break Even
        if progress := (abs(current_price - entry) / abs(tp - entry) if abs(tp - entry) > 0 else 0) >= 0.5:
            new_sl = entry
            if (direction == "LONG" and new_sl > sl) or (direction == "SHORT" and new_sl < sl):
                trade["sl"] = new_sl
                await send_alert(
                    f"""
🟡 BREAK EVEN

📊 {symbol}

━━━━━━━━━━━━━━

📌 Entry Protected

🛡 SL moved to Entry

💰 Risk-Free Trade

━━━━━━━━━━━━━━
"""
                )

        # Trailing Stop
        if progress >= 0.75:
            atr = trade.get("atr", abs(tp - entry) * 0.3)
            trail = atr * 0.5
            new_sl = current_price - trail if direction == "LONG" else current_price + trail
            if (direction == "LONG" and new_sl > sl) or (direction == "SHORT" and new_sl < sl):
                trade["sl"] = new_sl
                await send_alert(
                    f"""
🚀 TRAILING ACTIVE

📊 {symbol}

━━━━━━━━━━━━━━

🔒 Profit Locked

🛡 New SL: {new_sl:.6f}

📈 Runner Mode Enabled

━━━━━━━━━━━━━━
"""
                )

    save_open_trades(trades)
