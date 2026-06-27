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

    symbols = get_top_symbols(20)

    results = []

    for symbol in symbols:

        df = get_ohlcv(
            symbol,
            "5m",
            200
        )

        if df is None:
            continue

        signal = get_signal(df)

        if signal:

            results.append({
                "symbol": symbol,
                **signal
            })

    results.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    top3 = results[:3]

    for trade in top3:

        await send_alert(
            f"""
{trade['symbol']}
{trade['direction']}

Confidence: {trade['confidence']}%
"""
        )

async def startup():

    await send_alert(
        "🚀 SMC Whale AI Started"
    )

def heartbeat():

    print("Worker Alive")

def main():

    asyncio.run(startup())

    schedule.every(1).minutes.do(heartbeat)

    schedule.every(5).minutes.do(
        lambda: asyncio.run(scan())
    )

    while True:

        schedule.run_pending()

        time.sleep(5)

if __name__ == "__main__":
    main()
