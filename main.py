import asyncio
import schedule
import time
import logging
from datetime import datetime
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
    reset_daily_pnl
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

from pathlib import Path
import os

MODEL_PATH = "/app/data/models/xgboost_model.pkl"

logger.info(f"📁 STARTUP CHECK → Model file exists: {Path(MODEL_PATH).exists()}")

if os.path.exists("/app/data/models"):
    logger.info(f"📁 Files in /app/data/models/: {os.listdir('/app/data/models/')}")
else:
    logger.info("📁 /app/data/models/ folder does not exist yet")


async def scan():
    """
    Scan for signals every 5 minutes
    Rate limited to avoid API blocks
    """
    await monitor_trades()
    
    try:
        logger.info("🔍 Starting signal scan (5-minute interval)...")
        
        if not trading_allowed():
            logger.info("Daily loss limit reached. Trading paused for today.")
            return
        
        symbols = get_top_symbols(30)
        logger.info(f"Scanning {len(symbols)} quality coins")
        
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
                    logger.info(f"🟢 SIGNAL FOUND: {symbol} {signal['direction']} conf={signal.get('confidence', 0)}")
            
            except Exception as e:
                logger.exception(f"Symbol failed: {symbol} | {e}")
        
        logger.info(f"Total signals found: {len(results)}")
        
        if not results:
            logger.info("No valid signals this scan.")
            return
        
        results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        top3 = results[:3]
        
        for trade in top3:
            if trade_exists(trade["symbol"]):
                logger.info(f"Skipping duplicate trade: {trade['symbol']}")
                continue
            
            trade_no = next_trade_number()
            balance = get_balance()["balance"]
            
            trade_data = {
                "symbol": trade["symbol"],
                "direction": trade["direction"],
                "entry": float(trade["entry"]),
                "sl": float(trade["sl"]),
                "tp": float(trade["tp"]),
                "qty": float(trade["qty"]),
                "status": "OPEN",
                "trade_no": trade_no
            }
            
                        logger.info(f"🚀 Sending order to Bybit: {trade['symbol']}")

            order = await execute_trade(trade_data)

            if not order:
                logger.error(f"❌ Failed to execute order for {trade['symbol']}")
                continue

            logger.info(f"✅ Order placed successfully: {order}")

            add_trade(trade_data)

            await send_alert(
                f"""
🟢 <b>#{trade_no}</b>

<b>{trade['symbol']}</b>

📈 Direction: <b>{trade['direction']}</b>

📍 Entry: <b>${trade['entry']:.6f}</b>

🛑 Stop Loss: <b>${trade['sl']:.6f}</b>

🎯 Take Profit: <b>${trade['tp']:.6f}</b>

📦 Quantity: <b>{trade['qty']:.4f}</b>

⚡ Leverage: <b>10x</b>

🔥 Confidence: <b>{trade.get('confidence', 0)}/100</b>

💰 Balance: <b>${balance:.2f}</b>
"""
            )

            
await send_alert(
                f"""
🟢 <b>#{trade_no}</b>

<b>{trade['symbol']}</b>

📈 Direction: <b>{trade['direction']}</b>

📍 Entry: <b>${trade['entry']:.6f}</b>

🛑 Stop Loss: <b>${trade['sl']:.6f}</b>

🎯 Take Profit: <b>${trade['tp']:.6f}</b>

📦 Quantity: <b>{trade['qty']:.4f}</b>

⚡ Leverage: <b>10x</b>

🔥 Confidence: <b>{trade.get('confidence', 0)}/100</b>

💰 Balance: <b>${balance:.2f}</b>
"""
            )
        
        if len(get_trade_history()) >= 10:
            train_model_incremental()
    
    except Exception as e:
        logger.exception(f"SCAN FAILED: {e}")


async def run_monitor():
    """Monitor trades - runs independently every 35 seconds"""
    try:
        await monitor_trades()
    except Exception as e:
        logger.exception(f"Monitor failed: {e}")


async def startup():
    await send_alert("🚀 <b>SMC Whale (Paper Mode)</b>\n\nQuality coins only - No meme coins!")


def heartbeat():
    logger.info("💚 Worker Alive")


def daily_reset():
    """Reset daily PnL at midnight"""
    reset_daily_pnl()
    asyncio.run(send_alert("📅 <b>New trading day started!</b> Daily PnL reset."))


def run_scan_sync():
    """Wrapper to run async scan"""
    try:
        asyncio.run(scan())
    except Exception as e:
        logger.exception(f"Scan wrapper error: {e}")


def run_monitor_sync():
    """Wrapper to run async monitor - RUNS EVERY 35 SECONDS"""
    try:
        asyncio.run(run_monitor())
    except Exception as e:
        logger.exception(f"Monitor wrapper error: {e}")


def main():
    logger.info("🚀 Starting SMC Whale AI - PAPER Mode")
    logger.info("📊 Scanning 30 quality coins (no meme coins)")
    logger.info("⏱️  Scan interval: 5 minutes (rate limited)")
    logger.info("🔍 Monitor interval: 35 seconds (catch exits fast)")
    
    # Initial startup
    asyncio.run(startup())
    
    # Initial scan
    run_scan_sync()
    
    # Initial monitor
    run_monitor_sync()
    
    # ===== SCHEDULE JOBS =====
    # Heartbeat every minute
    schedule.every(1).minutes.do(heartbeat)
    
    # Monitor every 35 seconds (FAST - catch exits quickly!)
    schedule.every(35).seconds.do(run_monitor_sync)
    
    # Scan every 5 minutes (RATE LIMITED - avoid API blocks)
    schedule.every(5).minutes.do(run_scan_sync)
    
    # Daily reset at midnight
    schedule.every().day.at("00:00").do(daily_reset)
    
    logger.info("✅ Scheduler initialized")
    logger.info("Loss limit: -$15 (15% drawdown from $100)")
    logger.info("Profit cap: UNLIMITED ✅")
    logger.info("Daily reset: 00:00 UTC")
    
    # Main loop
    while True:
        try:
            schedule.run_pending()
            time.sleep(5)
        except Exception as e:
            logger.exception(f"Main loop error: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
