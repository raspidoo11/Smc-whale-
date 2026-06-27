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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)
exchange = get_exchange()


async def scan():
    """Main signal scanning + new trade alerts"""
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
                logger.info(f"Scanning {symbol}")

                df_15m = get_ohlcv(symbol, "15m", 200)
                df_5m = get_ohlcv(symbol, "5m", 200)

                if df_15m is None or df_5m is None:
                    logger.warning(f"{symbol} returned no data")
                    continue

                signal = get_signal(df_15m, df_5m)

                if signal:
                    qty = calculate_qty(signal["entry"], signal["sl"])
                    signal["qty"] = qty

                    logger.info(
                        f"SIGNAL {symbol} {signal['direction']} {signal.get('confidence', 0)}%"
                    )

                    results.append({
                        "symbol": symbol,
                        **signal
                    })

            except Exception as e:
                logger.exception(f"Symbol failed: {symbol} | {e}")

        logger.info(f"Scan complete. Signals found: {len(results)}")

        results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        top3 = results[:3]

        for trade in top3:
            add_trade({
                "symbol": trade["symbol"],
                "direction": trade["direction"],
                "entry": trade["entry"],
                "sl": trade["sl"],
                "tp": trade["tp"],
                "qty": trade["qty"],
                "status": "OPEN"
            })

            await send_alert(
                f"📈 NEW PAPER TRADE\n\n"
                f"{trade['symbol']}\n"
                f"Direction: {trade['direction']}\n"
                f"Entry: {trade['entry']:.4f}\n"
                f"SL: {trade['sl']:.4f}\n"
                f"TP: {trade['tp']:.4f}\n"
                f"Qty: {trade['qty']}\n"
                f"Confidence: {trade.get('confidence', 0)}%"
            )

        # Retrain XGBoost if enough data
        if len(get_trade_history()) >= 10:
            train_model()

    except Exception as e:
        logger.exception(f"SCAN FAILED: {e}")


async def run_monitor():
    """Fast monitoring loop for open trades"""
    try:
        await monitor_trades()
    except Exception as e:
        logger.exception(f"Monitor failed: {e}")


async def startup():
    await send_alert("🚀 SMC Whale AI Started")


def heartbeat():
    logger.info("Worker Alive")


def run_scan_sync():
    asyncio.run(scan())


def run_monitor_sync():
    asyncio.run(run_monitor())


def main():
    logger.info("🚀 Starting SMC Whale AI")

    asyncio.run(startup())
    logger.info("Running initial scan + monitor")
    run_scan_sync()
    run_monitor_sync()

    # Heartbeat every minute
    schedule.every(1).minutes.do(heartbeat)

    # Fast monitoring every 45 seconds (syncs much better with live price)
    schedule.every(45).seconds.do(run_monitor_sync)

    # Signal scanning every 2 minutes
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
