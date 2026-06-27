import asyncio
import schedule
import time
import logging

from scanner import get_top_symbols, get_ohlcv
from strategy import get_signal
from paper_trader import calculate_qty
from exchange import get_exchange
from telegram_alerts import send_alert
from trade_manager import add_trade, trading_allowed
from trade_monitor import monitor_trades
from xgboost_trainer import train_model
from trade_manager import get_trade_history
from demo_executor import execute_trade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)
exchange = get_exchange()


async def scan():
    await monitor_trades()

    try:
        logger.info("Starting signal scan...")

        if not trading_allowed():
            logger.info("Daily target reached. Trading paused.")
            return

        symbols = get_top_symbols(20)
        logger.info(f"Found {len(symbols)} symbols")

        results = []

        for symbol in symbols:
            try:
                df_15m = get_ohlcv(symbol, "15m", 200)
                df_5m = get_ohlcv(symbol, "5m", 200)

                if df_15m is None or df_5m is None:
                    continue

                signal = get_signal(df_15m, df_5m)

                if signal:
                    qty = calculate_qty(signal["entry"], signal["sl"])
                    signal["qty"] = qty
                    results.append({"symbol": symbol, **signal})
                    logger.info(f"SIGNAL FOUND: {symbol} {signal['direction']} conf={signal.get('confidence', 0)}")

            except Exception as e:
                logger.exception(f"Symbol failed: {symbol} | {e}")

        logger.info(f"Total signals found: {len(results)}")

        if not results:
            logger.info("No valid signals this scan.")
            return

        results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        top3 = results[:3]

        logger.info(f"Processing top {len(top3)} signals...")

        for trade in top3:
            order = await execute_trade(trade)

            if order:
                add_trade({
                    "symbol": trade["symbol"],
                    "direction": trade["direction"],
                    "entry": trade["entry"],
                    "sl": trade["sl"],
                    "tp": trade["tp"],
                    "qty": trade["qty"],
                    "status": "OPEN",
                    "order_id": order.get("id")
                })

                await send_alert(
                    f"🚀 DEMO TRADE EXECUTED\n\n"
                    f"{trade['symbol']}\n"
                    f"Direction: {trade['direction']}\n"
                    f"Entry: {trade['entry']:.4f}\n"
                    f"SL: {trade['sl']:.4f}\n"
                    f"TP: {trade['tp']:.4f}\n"
                    f"Qty: {trade['qty']}\n"
                    f"Confidence: {trade.get('confidence', 0)}%"
                )
                logger.info(f"TRADE SENT: {trade['symbol']} {trade['direction']}")
            else:
                logger.warning(f"Trade execution FAILED for {trade['symbol']}")

        if len(get_trade_history()) >= 10:
            train_model()

    except Exception as e:
        logger.exception(f"SCAN FAILED: {e}")


async def run_monitor():
    try:
        await monitor_trades()
    except Exception as e:
        logger.exception(f"Monitor failed: {e}")


async def startup():
    await send_alert("🚀 SMC Whale AI Started (Demo Mode)")


def heartbeat():
    logger.info("Worker Alive")


def run_scan_sync():
    asyncio.run(scan())


def run_monitor_sync():
    asyncio.run(run_monitor())


def main():
    logger.info("🚀 Starting SMC Whale AI - DEMO Mode")

    asyncio.run(startup())
    run_scan_sync()
    run_monitor_sync()

    schedule.every(1).minutes.do(heartbeat)
    schedule.every(45).seconds.do(run_monitor_sync)
    schedule.every(2).minutes.do(run_scan_sync)

    while True:
        try:
            schedule.run_pending()
            time.sleep(5)
        except Exception as e:
            logger.exception(f"Main loop error: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
