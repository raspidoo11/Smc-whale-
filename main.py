import asyncio
import schedule
import time
import logging
from datetime import datetime, timezone
from scanner import get_live_symbols as get_top_symbols, get_ohlcv
from strategy import get_signal
from paper_trader import calculate_qty
from bybit_executor import execute_trade
from exchange import get_exchange
from telegram_alerts import send_alert
from trade_manager import (
    add_trade,
    trading_allowed,
    trade_exists,
    next_trade_number,
    get_balance,
    reset_daily_pnl,
    get_open_trades,
    get_signal_hash_exists,
    save_signal_hash
)
from trade_monitor import monitor_trades
from xgboost_trainer import train_model_incremental
from trade_manager import get_trade_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

exchange = get_exchange()

# ==================== CONFIG ====================
MAX_OPEN_TRADES = 10
SCAN_LOCK = asyncio.Lock()           # Prevents overlapping scans


async def get_fresh_symbols(limit=30):
    """Get symbols excluding those we already have open trades on"""
    open_trades = get_open_trades()
    open_symbols = {t["symbol"] for t in open_trades if t.get("status") == "OPEN"}

    # Get more symbols than needed so we can filter
    all_symbols = get_top_symbols(100)

    fresh_symbols = [s for s in all_symbols if s not in open_symbols][:limit]
    logger.info(f"📊 Fresh symbols selected: {len(fresh_symbols)} (excluded {len(open_symbols)} open)")

    return fresh_symbols


async def scan():
    """Main scanning function with lock + fresh symbols"""
    async with SCAN_LOCK:   # Prevent overlapping scans
        await monitor_trades()

        try:
            logger.info("🔍 Starting signal scan...")

            if not trading_allowed():
                logger.info("⛔ Daily loss limit reached. Trading paused.")
                return

            open_trades = get_open_trades()
            if len([t for t in open_trades if t.get("status") == "OPEN"]) >= MAX_OPEN_TRADES:
                logger.info(f"📉 Max open trades ({MAX_OPEN_TRADES}) reached. Skipping scan.")
                return

            symbols = await get_fresh_symbols(30)

            results = []

            for symbol in symbols:
                try:
                    df_15m = get_ohlcv(symbol, "15m", 200)
                    df_5m = get_ohlcv(symbol, "5m", 200)

                    if df_15m is None or df_5m is None:
                        continue

                    signal = get_signal(df_15m, df_5m)

                    if signal:
                        # Check for duplicate using signal_hash
                        if get_signal_hash_exists(signal.get("signal_hash")):
                            logger.info(f"⏭️ Duplicate signal_hash skipped: {symbol}")
                            continue

                        qty = calculate_qty(signal["entry"], signal["sl"])
                        signal["qty"] = qty
                        results.append({"symbol": symbol, **signal})

                except Exception as e:
                    logger.exception(f"Symbol scan error: {symbol} | {e}")

            if not results:
                logger.info("No valid signals this scan.")
                return

            # Rank by confidence
            results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            top_signals = results[:3]

            for trade in top_signals:
                if len([t for t in get_open_trades() if t.get("status") == "OPEN"]) >= MAX_OPEN_TRADES:
                    break

                trade_no = next_trade_number()

                # Build trade data (preserve everything from strategy)
                trade_data = {
                    **trade,
                    "entry": float(trade["entry"]),
                    "sl": float(trade["sl"]),
                    "tp": float(trade["tp"]),
                    "qty": float(trade["qty"]),
                    "status": "OPEN",
                    "trade_no": trade_no,
                }

                order = await execute_trade(trade_data)

                if not order:
                    logger.error(f"❌ Failed to execute: {trade['symbol']}")
                    continue

                add_trade(trade_data)

                # Save signal_hash to prevent duplicates
                if trade.get("signal_hash"):
                    save_signal_hash(trade["signal_hash"])

                # ==================== IMPROVED TELEGRAM ALERT ====================
                direction = trade.get('direction', 'LONG')
                is_long = direction == "LONG"
                dir_emoji = "🟢" if is_long else "🔴"

                alert_text = f"""
{dir_emoji} <b>#{trade_no}</b> | {trade['symbol']} {dir_emoji} <b>{direction}</b>

💰 <b>Entry</b>:   <code>${trade['entry']:.6f}</code>

❌ <b>SL</b>:       <code>${trade['sl']:.6f}</code>

✅ <b>TP</b>:       <code>${trade['tp']:.6f}</code>

🗑️ <b>Qty</b>:      <code>{trade['qty']:.4f}</code>

🔥 <b>Confidence</b>: <b>{trade.get('confidence', 0)}/100</b>
"""

                await send_alert(alert_text.strip())

            # Retrain model periodically
            if len(get_trade_history()) >= 10:
                train_model_incremental()

        except Exception as e:
            logger.exception(f"SCAN FAILED: {e}")


# ==================== SCHEDULER WRAPPERS ====================

async def run_monitor():
    try:
        await monitor_trades()
    except Exception as e:
        logger.exception(f"Monitor error: {e}")


def run_scan_sync():
    try:
        asyncio.run(scan())
    except Exception as e:
        logger.exception(f"Scan wrapper error: {e}")


def run_monitor_sync():
    try:
        asyncio.run(run_monitor())
    except Exception as e:
        logger.exception(f"Monitor wrapper error: {e}")


def heartbeat():
    logger.info("💚 Worker alive")


def daily_reset():
    reset_daily_pnl()
    asyncio.run(send_alert("📅 New trading day started. Daily PnL reset."))


async def startup():
    await send_alert("🚀 <b>SMC Whale Bot Started</b>\nPaper Mode + Session Bonus Active")


def main():
    logger.info("🚀 Starting SMC Whale AI (Improved Version)")
    logger.info(f"📊 Max Open Trades: {MAX_OPEN_TRADES}")
    logger.info("🔒 Using asyncio.Lock() for scan protection")

    asyncio.run(startup())
    run_scan_sync()
    run_monitor_sync()

    schedule.every(1).minutes.do(heartbeat)
    schedule.every(35).seconds.do(run_monitor_sync)
    schedule.every(5).minutes.do(run_scan_sync)
    schedule.every().day.at("00:00").do(daily_reset)

    logger.info("✅ Scheduler started")

    while True:
        try:
            schedule.run_pending()
            time.sleep(5)
        except Exception as e:
            logger.exception(f"Main loop error: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
