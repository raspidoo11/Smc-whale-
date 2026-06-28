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
                f"""
🟢 PAPER OPENED

📊 {trade['symbol']}

━━━━━━━━━━━━━━

🎯 Direction: {trade['direction']}

📌 Entry: {trade['entry']:.6f}

🛑 Stop Loss: {trade['sl']:.6f}

🎯 Take Profit: {trade['tp']:.6f}

📦 Qty: {trade['qty']}

🔥 Confidence: {trade.get('confidence', 0)}/100

💰 Risk: 5% Balance

━━━━━━━━━━━━━━

🤖 SMC Whale AI
📝 Paper Trade
"""
            )

        if len(get_trade_history()) >= 10:
            train_model()

    except Exception as e:
        logger.exception(f"SCAN FAILED: {e}")


async def run_monitor():
    try:
        await monitor_trades()
    except Exception as e:
        logger.exception(f"Monitor failed: {e}")


from trade_manager import get_balance

async def startup():

    balance = get_balance()["balance"]

    await send_alert(
        f"""
<b>🚀 SMC WHALE AI ONLINE</b>

━━━━━━━━━━━━━━

💰 Balance: ${balance:.2f}

📊 Market: Bybit Futures

🧠 Strategy: SMC + XGBoost

⏱ Scan: Every 2 Minutes

━━━━━━━━━━━━━━

✅ Telegram Connected
✅ Scanner Ready
✅ Paper Trading Active
"""
    )

def heartbeat():
    logger.info("Worker Alive")


def run_scan_sync():
    asyncio.run(scan())


def run_monitor_sync():
    asyncio.run(run_monitor())


def main():
    logger.info("🚀 Starting SMC Whale AI - PAPER Mode")

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
