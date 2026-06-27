import asyncio
import schedule
import time
import logging
from scanner import get_top_symbols, get_ohlcv
from strategy import get_signal
from paper_trader import calculate_qty
from exchange import get_exchange
from telegram_alerts import send_alert
logging.basicConfig(
level=logging.INFO,
format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(name)
exchange = get_exchange()
async def scan():
    try:
    logger.info("Starting scan...")
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
                qty = calculate_qty(
                    signal["entry"],
                    signal["sl"]
                )

                signal["qty"] = qty

                logger.info(
                    f"SIGNAL {symbol} "
                    f"{signal['direction']} "
                    f"{signal['confidence']}%"
                )

                results.append({
                    "symbol": symbol,
                    **signal
                })

        except Exception as e:
            logger.exception(
                f"Symbol failed: {symbol} | {e}"
            )

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
            f"📈 {trade['symbol']}\n\n"
            f"Direction: {trade['direction']}\n"
            f"Entry: {trade['entry']:.4f}\n"
            f"SL: {trade['sl']:.4f}\n"
            f"TP: {trade['tp']:.4f}\n"
            f"Qty: {trade['qty']}\n"
            f"Confidence: {trade['confidence']}%"
        )

except Exception as e:
    logger.exception(f"SCAN FAILED: {e}")
async def startup():
await send_alert(
"🚀 SMC Whale AI Started"
)
def heartbeat():
logger.info("Worker Alive")
def run_scan():
try:
asyncio.run(scan())
except Exception as e:
logger.exception(
f"Scheduled scan failed: {e}"
)
def main():
logger.info(
"🚀 Starting SMC Whale AI"
)
asyncio.run(startup())

logger.info(
    "Running initial scan"
)

run_scan()

schedule.every(1).minutes.do(
    heartbeat
)

schedule.every(5).minutes.do(
    run_scan
)

while True:
    try:
        schedule.run_pending()
        time.sleep(5)

    except Exception as e:
        logger.exception(
            f"Main loop error: {e}"
        )

        time.sleep(30)
if name == "main":
main()
