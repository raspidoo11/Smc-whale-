import asyncio
import logging
from exchange import get_exchange
from trade_manager import get_open_trades, close_trade, update_balance
from telegram_alerts import send_alert
import pandas as pd

logger = logging.getLogger(__name__)
exchange = get_exchange()


async def get_current_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        logger.error(f"Price fetch failed for {symbol}: {e}")
        return None


async def monitor_trades():
    """Monitor all open trades for TP/SL, BE, and trailing"""
    trades = get_open_trades()
    if not trades:
        return

    logger.info(f"Monitoring {len(trades)} open trades...")

    for trade in trades[:]:  # copy to avoid modification issues
        if trade.get("status") != "OPEN":
            continue

        symbol = trade["symbol"]
        entry = trade["entry"]
        sl = trade["sl"]
        tp = trade["tp"]
        direction = trade["direction"]

        current_price = await get_current_price(symbol)
        if not current_price:
            continue

        pnl = (current_price - entry) if direction == "LONG" else (entry - current_price)
        atr = trade.get("atr", abs(tp - entry) * 0.5)  # fallback

        # TP Hit
        if (direction == "LONG" and current_price >= tp) or (direction == "SHORT" and current_price <= tp):
            result = "WIN"
            close_trade(symbol, current_price, result)
            update_balance(pnl * trade.get("qty", 1))
            await send_alert(
                f"✅ PAPER WIN\n\n"
                f"{symbol}\n"
                f"Entry: {entry:.4f}\n"
                f"Exit: {current_price:.4f}\n"
                f"Profit: +${pnl*trade.get('qty',1):.2f}\n"
                f"Balance: ${get_balance()['balance']:.2f}"
            )
            continue

        # SL Hit
        if (direction == "LONG" and current_price <= sl) or (direction == "SHORT" and current_price >= sl):
            result = "LOSS"
            close_trade(symbol, current_price, result)
            update_balance(pnl * trade.get("qty", 1))
            await send_alert(
                f"❌ PAPER LOSS\n\n"
                f"{symbol}\n"
                f"Entry: {entry:.4f}\n"
                f"Exit: {current_price:.4f}\n"
                f"Loss: -${abs(pnl*trade.get('qty',1)):.2f}\n"
                f"Balance: ${get_balance()['balance']:.2f}"
            )
            continue

        # Break Even (50% to TP)
        distance_to_tp = abs(tp - entry)
        if abs(current_price - entry) >= 0.5 * distance_to_tp:
            new_sl = entry  # move to breakeven
            if new_sl != sl:
                # Update in memory (you may want to persist this)
                trade["sl"] = new_sl
                logger.info(f"BE triggered for {symbol} - SL moved to {new_sl}")

        # Trailing Stop (75% to TP)
        if abs(current_price - entry) >= 0.75 * distance_to_tp:
            trail = atr * 0.5
            new_sl = current_price - trail if direction == "LONG" else current_price + trail
            if (direction == "LONG" and new_sl > sl) or (direction == "SHORT" and new_sl < sl):
                trade["sl"] = new_sl
                logger.info(f"Trailing SL updated for {symbol} to {new_sl}")

    # Save updated trades if you modified SLs
    # save_open_trades(trades)  # uncomment if persisting SL changes
