import asyncio
import schedule
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)
from scanner import get_top_symbols, get_ohlcv
from strategy import get_signal

from exchange import get_exchange
from telegram_alerts import send_alert

exchange = get_exchange()

async def scan():

    logger.info("Starting scan...")

    symbols = get_top_symbols(20)

    logger.info(f"Found {len(symbols)} symbols")

    results = []

    for symbol in symbols:

        logger.info(f"Scanning {symbol}")

        df = get_ohlcv(symbol, "5m", 200)

        if df is None:

            logger.warning(f"{symbol} returned no data")

            continue

        signal = get_signal(df)

        if signal:

            logger.info(
                f"SIGNAL {symbol} "
                f"{signal['direction']} "
                f"{signal['confidence']}%"
            )

            results.append({
                "symbol": symbol,
                **signal
            })

    logger.info(
        f"Scan complete. Signals found: {len(results)}"
    )

    results.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    top3 = results[:3]

    logger.info(
        f"Sending {len(top3)} Telegram alerts"
    )

    for trade in top3:

        await send_alert(
            f"{trade['symbol']}\n"
            f"{trade['direction']}\n"
            f"Confidence: {trade['confidence']}%"
        )

async def startup():

    await send_alert(
        "🚀 SMC Whale AI Started"
    )

def heartbeat():

    print("Worker Alive")

def main():

    asyncio.run(startup())

    logger.info("Running initial scan")

    asyncio.run(scan())

    schedule.every(1).minutes.do(heartbeat)

    schedule.every(5).minutes.do(
        lambda: asyncio.run(scan())
    )

    while True:

        schedule.run_pending()

        time.sleep(5)
