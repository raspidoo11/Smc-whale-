import logging
import asyncio
from trade_manager import get_open_trades, get_balance
from exchange import get_exchange
from bybit_executor import activate_trailing_stop
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

                if hit_sl:
                    # Close on Stop Loss
                    exit_price = sl
                    exit_reason = "Stop Loss Hit"

                    logger.info(f"🚨 {exit_reason} on {symbol} at ${exit_price:.6f}")

                    # For now we still use paper close for demo. 
                    # You can replace with live close later.
                    from paper_trader import close_paper_trade_with_fees
                    pnl_after_fees = close_paper_trade_with_fees(trade, exit_price, exit_reason)

                    balance = get_balance()["balance"]
                    await send_alert(
                        f"❌ {exit_reason}\n"
                        f"{direction} {symbol}\n"
                        f"Entry: ${entry:.6f} → Exit: ${exit_price:.6f}\n"
                        f"Qty: {qty}\n"
                        f"Net PnL: ${pnl_after_fees:.2f}\n"
                        f"Balance: ${balance:.2f}"
                    )
                    train_model_incremental()
                    continue

                if hit_tp:
                    # Activate trailing stop instead of closing
                    logger.info(f"🚀 Activating trailing stop on {symbol} (TP reached)")

                    result = await activate_trailing_stop(
                        symbol=symbol,
                        direction=direction,
                        qty=qty,
                        trail_percent=0.5
                    )

                    if result:
                        await send_alert(
                            f"🚀 Trailing Stop Activated\n"
                            f"{direction} {symbol}\n"
                            f"Original TP: ${tp:.6f}\n"
                            f"Current Price: ${current_price:.6f}\n"
                            f"Trail: 0.5%"
                        )
                    else:
                        logger.error(f"Failed to activate trailing stop for {symbol}")

                    train_model_incremental()
                    continue

            except Exception as e:
                logger.exception(f"❌ Error monitoring trade {trade.get('symbol')}: {e}")

    except Exception as e:
        logger.exception(f"❌ Monitor trades failed: {e}")


async def main():
    await monitor_trades()


if __name__ == "__main__":
    asyncio.run(main())
