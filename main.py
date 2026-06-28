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
    get_balance
)
from trade_monitor import monitor_trades
from xgboost_trainer import train_model
from trade_manager import get_trade_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

exchange = get_exchange()

# Global event loop for scheduling
loop = None

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
            if trade_exists(trade["symbol"]):
                logger.info(f"Skipping duplicate trade: {trade['symbol']}")
                continue
            
            trade_no = next_trade_number()
            balance = get_balance()["balance"]
            
            # FIXED: Store trade_no in the trade dict for monitoring
            trade_data = {
                "symbol": trade["symbol"],
                "direction": trade["direction"],
                "entry": float(trade["entry"]),
                "sl": float(trade["sl"]),
                "tp": float(trade["tp"]),
                "qty": float(trade["qty"]),
                "status": "OPEN",
                "trade_no": trade_no  # NEW: Store trade number for alerts
            }
            
            add_trade(trade_data)
            
            await send_alert(
                f"""
🟢 #{trade_no}

{trade['symbol']}

📈 {trade['direction']}

📍 {trade['entry']:.6f}

🛑 {trade['sl']:.6f}

🎯 {trade['tp']:.6f}

📦 {trade['qty']:.4f}

⚡ 10x

🔥 {trade.get('confidence', 0)}/100

💰 ${balance:.2f}
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

async def startup():
    await send_alert("🚀 SMC Whale AI Started (Paper Mode)")

def heartbeat():
    logger.info("Worker Alive")

def run_scan_sync():
    """Wrapper to run async scan"""
    try:
        asyncio.run(scan())
    except Exception as e:
        logger.exception(f"Scan wrapper error: {e}")

def run_monitor_sync():
    """Wrapper to run async monitor"""
    try:
        asyncio.run(run_monitor())
    except Exception as e:
        logger.exception(f"Monitor wrapper error: {e}")

def main():
    logger.info("🚀 Starting SMC Whale AI - PAPER Mode")
    
    # Initial startup
    asyncio.run(startup())
    
    # Initial scan
    run_scan_sync()
    
    # Initial monitor
    run_monitor_sync()
    
    # Schedule jobs
    schedule.every(1).minutes.do(heartbeat)
    schedule.every(45).seconds.do(run_monitor_sync)
    schedule.every(2).minutes.do(run_scan_sync)
    
    logger.info("Scheduler initialized")
    
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
