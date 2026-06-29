import logging
import asyncio
from exchange import get_exchange
from trade_manager import (
    get_open_trades,
    save_open_trades,
    close_trade,
    update_balance,
    get_balance
)
from telegram_alerts import send_alert
from xgboost_trainer import train_model_incremental

logger = logging.getLogger(__name__)

exchange = get_exchange()

async def get_current_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return float(ticker.get("last", 0))
    except Exception as e:
        logger.error(f"Price error {symbol}: {e}")
        return None

async def monitor_trades():
    trades = get_open_trades()
    
    if not trades:
        return
    
    modified = False
    logger.info(f"Monitoring {len(trades)} trades")
    
    closed_indices = []
    
    for idx, trade in enumerate(trades):
        if trade.get("status") != "OPEN":
            continue
        
        symbol = trade["symbol"]
        entry = float(trade["entry"])
        sl = float(trade["sl"])
        tp = float(trade["tp"])
        qty = float(trade.get("qty", 1))
        direction = trade["direction"]
        
        trade_no = trade.get("trade_no", idx + 1)
        
        current_price = await get_current_price(symbol)
        
        if not current_price:
            logger.warning(f"Failed to get price for {symbol}")
            continue
        
        if not trade.get("trail_active", False):
            if direction == "LONG":
                trigger = entry + ((tp - entry) * 0.75)
                
                if current_price >= trigger:
                    trade["trail_active"] = True
                    modified = True
                    
                    await send_alert(
                        f"🚀 <b>#{trade_no} - Trailing Stop Activated</b>\n\n"
                        f"<b>{symbol}</b>\n\n"
                        f"Now following price upward at 0.5% trail"
                    )
            else:
                trigger = entry - ((entry - tp) * 0.75)
                
                if current_price <= trigger:
                    trade["trail_active"] = True
                    modified = True
                    
                    await send_alert(
                        f"🚀 <b>#{trade_no} - Trailing Stop Activated</b>\n\n"
                        f"<b>{symbol}</b>\n\n"
                        f"Now following price downward at 0.5% trail"
                    )
        
        if trade.get("trail_active", False):
            if direction == "LONG":
                new_sl = current_price * 0.995
                
                if new_sl > trade["sl"]:
                    old_sl = trade["sl"]
                    trade["sl"] = new_sl
                    modified = True
                    logger.info(f"TRAIL UPDATE: {symbol} SL ${old_sl:.6f} → ${new_sl:.6f}")
            
            else:
                new_sl = current_price * 1.005
                
                if new_sl < trade["sl"]:
                    old_sl = trade["sl"]
                    trade["sl"] = new_sl
                    modified = True
                    logger.info(f"TRAIL UPDATE: {symbol} SL ${old_sl:.6f} → ${new_sl:.6f}")
        
        sl = float(trade["sl"])
        
        if direction == "LONG":
            pnl = (current_price - entry) * qty
            hit_tp = current_price >= tp
            hit_sl = current_price <= sl
        
        else:
            pnl = (entry - current_price) * qty
            hit_tp = current_price <= tp
            hit_sl = current_price >= sl
        
        if hit_tp or hit_sl:
            result = "WIN" if hit_tp else "LOSS"
            
            close_trade(symbol, current_price, result)
            update_balance(pnl)
            
            balance = get_balance()["balance"]
            
            status_emoji = "✅" if hit_tp else "❌"
            exit_type = "Take Profit Hit" if hit_tp else "Stop Loss Hit"
            pnl_sign = "+" if pnl >= 0 else ""
            
            await send_alert(
                f"{status_emoji} <b>#{trade_no} - {exit_type}</b>\n\n"
                f"<b>{symbol}</b>\n\n"
                f"Direction: <b>{direction}</b>\n\n"
                f"📍 Entry: <b>${entry:.6f}</b>\n"
                f"🚪 Exit: <b>${current_price:.6f}</b>\n\n"
                f"💹 Profit/Loss: <b>{pnl_sign}${pnl:.2f}</b>\n"
                f"💰 Balance: <b>${balance:.2f}</b>"
            )
            
            logger.info(f"\n🤖 LEARNING FROM TRADE #{trade_no}...")
            train_model_incremental()
            logger.info(f"✅ Model updated with new trade data\n")
            
            closed_indices.append(idx)
            modified = True
    
    if modified:
        updated_trades = [t for i, t in enumerate(trades) if i not in closed_indices]
        save_open_trades(updated_trades)
        logger.info(f"Trades updated. Remaining open: {len(updated_trades)}")
