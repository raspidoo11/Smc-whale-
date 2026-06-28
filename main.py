import asyncio
import schedule
import time
import logging

from scanner import get_top_symbols, get_ohlcv
from strategy import get_signal
from paper_trader import calculate_qty
from exchange import get_exchange
from telegram_alerts import send_alert
from trade_manager import (
    add_trade,
    trading_allowed,
    trade_exists,
    next_trade_number,
    get_balance,
    get_trade_history
)
from trade_monitor import monitor_trades
from xgboost_trainer import train_model

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
            logger.info(
                "Daily target reached. Trading paused."
            )

            return

        symbols = get_top_symbols(20)

        logger.info(
            f"Found {len(symbols)} symbols"
        )

        results = []

        for symbol in symbols:

            try:

                df_15m = get_ohlcv(
                    symbol,
                    "15m",
                    200
                )

                df_5m = get_ohlcv(
                    symbol,
                    "5m",
                    200
                )

                if (
                    df_15m is None
                    or df_5m is None
                ):
                    continue

                signal = get_signal(
                    df_15m,
                    df_5m
                )

                if signal:

                    qty = calculate_qty(
                        signal["entry"],
                        signal["sl"]
                    )

                    signal["qty"] = qty

                    results.append({
                        "symbol": symbol,
                        **signal
                    })

                    logger.info(
                        f"SIGNAL FOUND: "
                        f"{symbol} "
                        f"{signal['direction']} "
                        f"conf="
                        f"{signal.get('confidence', 0)}"
                    )

            except Exception as e:

                logger.exception(
                    f"Symbol failed: "
                    f"{symbol} | {e}"
                )

        logger.info(
            f"Total signals found: "
            f"{len(results)}"
        )

        if not results:

            logger.info(
                "No valid signals this scan."
            )

            return

        results.sort(
            key=lambda x: x.get(
                "confidence",
                0
            ),
            reverse=True
        )

        top3 = results[:3]

        for trade in top3:

            if trade_exists(
                trade["symbol"]
            ):

                logger.info(
                    f"Duplicate skipped: "
                    f"{trade['symbol']}"
                )

                continue

            trade_no = (
                next_trade_number()
            )

            balance = (
                get_balance()[
                    "balance"
                ]
            )

            add_trade({

                "trade_no": trade_no,

                "symbol":
                    trade["symbol"],

                "direction":
                    trade["direction"],

                "entry":
                    float(
                        trade["entry"]
                    ),

                "sl":
                    float(
                        trade["sl"]
                    ),

                "tp":
                    float(
                        trade["tp"]
                    ),

                "qty":
                    float(
                        trade["qty"]
                    ),

                "status":
                    "OPEN",

                "be_active":
                    False,

                "trail_active":
                    False

            })

            await send_alert(

                f"""
🟢 #{trade_no}

{trade['symbol']}

📈 {trade['direction']}

📍 {trade['entry']:.6f}
🛑 {trade['sl']:.6f}
🎯 {trade['tp']:.6f}

📦 {trade['qty']}
🔥 {trade.get('confidence',0)}/100

💰 ${balance:.2f}
"""
            )

            logger.info(
                f"Trade #{trade_no} opened"
            )

        if len(
            get_trade_history()
        ) >= 10:

            logger.info(
                "Retraining XGBoost..."
            )

            train_model()

    except Exception as e:

        logger.exception(
            f"SCAN FAILED: {e}"
        )


async def run_monitor():
    try:
        await monitor_trades()
    except Exception as e:
        logger.exception(
            f"Monitor failed: {e}"
        )


async def startup():
    balance = (
        get_balance()["balance"]
    )

    await send_alert(
        f"""
🚀 PAPER TRADER ONLINE
💰 Balance
${balance:.2f}
📊 Bybit Futures
Top 20 Symbols
⚡ Scan: 2 Minutes
🎯 Max Signals: 3
"""
    )


def heartbeat():
    logger.info(
        "Worker Alive"
    )


def run_scan_sync():
    asyncio.run(
        scan()
    )


def run_monitor_sync():
    asyncio.run(
        run_monitor()
    )


def main():
    logger.info(
        "Starting Paper Trader..."
    )

    asyncio.run(
        startup()
    )

    run_scan_sync()

    run_monitor_sync()

    schedule.every(
        1
    ).minutes.do(
        heartbeat
    )

    schedule.every(
        45
    ).seconds.do(
        run_monitor_sync
    )

    schedule.every(
        2
    ).minutes.do(
        run_scan_sync
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


if __name__ == "__main__":
    main()
