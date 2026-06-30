import logging
import asyncio
from trade_manager import get_open_trades, get_balance
from exchange import get_exchange
from paper_trader import close_paper_trade_with_fees
from telegram_alerts import send_alert
from xgboost_trainer import train_model_incremental

logger = logging.getLogger(__name__)
exchange = get_exchange()


async def get_current_price(symbol):
    try:
        if not exchange:
            return None
        ticker = exchange.fetch_ticker(symbol)
        return ticker.get("last")
    except Exception as e:
        logger.debug(f"Failed to get price for {symbol}: {e}")
        return None


async def monitor_trades():
    try:
        open_trades = get_open_trades()
        logger.info(f"📊 Monitoring {len(open_trades)} open trade(s)")

        if not open_trades:
            return

        for trade in open_trades:
            try:
                symbol = trade.get("symbol")
                if not symbol:
                    continue

                current_price = await get_current_price(symbol)
                if current_price is None:
                    continue

                entry = float(trade.get("entry", 0))
                sl = float(trade.get("sl", 0))
                tp = float(trade.get("tp", 0))
                direction = trade.get("direction")
                qty = float(trade.get("qty", 0))

                if not all([entry, sl, tp, direction, qty]):
                    continue

                logger.info(
                    f"{symbol} | Current={current_price:.6f} | TP={tp:.6f} | SL={sl:.6f}"
                )

                if direction == "LONG":
                    hit_tp = current_price >= tp
                    hit_sl = current_price <= sl
                elif direction == "SHORT":
                    hit_tp = current_price <= tp
                    hit_sl = current_price >= sl
                else:
                    continue

                if not (hit_tp or hit_sl):
                    continue

                exit_price = tp if hit_tp else sl
                exit_reason = "Take Profit Hit" if hit_tp else "Stop Loss Hit"

                logger.info(f"🚨 {exit_reason} on {symbol} at ${exit_price:.6f}")

                pnl_after_fees = close_paper_trade_with_fees(
                    trade,
                    exit_price,
                    exit_reason
                )

                balance = get_balance()["balance"]
                status_emoji = "✅" if hit_tp else "❌"

                await send_alert(
                    f"{status_emoji} {exit_reason}\n"
                    f"{direction} {symbol}\n"
                    f"Entry: ${entry:.6f} → Exit: ${exit_price:.6f}\n"
                    f"Qty: {qty}\n"
                    f"Net PnL: ${pnl_after_fees:.2f}\n"
                    f"Balance: ${balance:.2f}"
                )

                train_model_incremental()

            except Exception as e:
                logger.exception(f"❌ Error monitoring trade {trade.get('symbol')}: {e}")

    except Exception as e:
        logger.exception(f"❌ Monitor trades failed: {e}")


async def main():
    await monitor_trades()


if __name__ == "__main__":
    asyncio.run(main())
