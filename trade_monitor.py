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
        confidence = trade.get("confidence", 70)

        current_price = await get_current_price(symbol)
        if not current_price:
            continue

        distance_to_tp = abs(tp - entry)
        progress = abs(current_price - entry) / distance_to_tp if distance_to_tp > 0 else 0

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
            await close_position(symbol, direction)  # if using executor
            close_trade(symbol, current_price, result)
            update_balance(pnl)
            bal = get_balance()["balance"]

            emoji = "✅" if hit_tp else "❌"
            await send_alert(
                f"{emoji} PAPER {result}\n\n"
                f"{symbol}\n"
                f"Entry: {entry:.4f}\n"
                f"Exit: {current_price:.4f}\n"
                f"Profit/Loss: {'+' if hit_tp else '-'}${abs(pnl):.2f}\n"
                f"Balance: ${bal:.2f}"
            )
            continue

        # Break-Even
        if confidence < 70 and progress >= 0.5:
            new_sl = entry
            if (direction == "LONG" and new_sl > sl) or (direction == "SHORT" and new_sl < sl):
                trade["sl"] = new_sl
                await send_alert(f"🔄 BE Triggered {symbol} - SL to entry")

        # Trailing Stop
        if progress >= 0.75:
            atr = trade.get("atr", distance_to_tp * 0.3)
            trail = atr * 0.5
            new_sl = current_price - trail if direction == "LONG" else current_price + trail
            if (direction == "LONG" and new_sl > sl) or (direction == "SHORT" and new_sl < sl):
                trade["sl"] = new_sl
                await send_alert(f"📈 Trailing SL {symbol} → {new_sl:.4f}")

    save_open_trades(trades)
