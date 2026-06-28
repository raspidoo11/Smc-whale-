import logging
from exchange import get_exchange
from trade_manager import (
    get_open_trades,
    save_open_trades,
    close_trade,
    update_balance,
    get_balance
)
from telegram_alerts import send_alert

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
    
    # Track indices to avoid index shifting during iteration
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
        
        # FIXED: Use trade_no from trade dict, fallback to index if not present
        trade_no = trade.get("trade_no", idx + 1)
        
        current_price = await get_current_price(symbol)
        
        if not current_price:
            logger.warning(f"Failed to get price for {symbol}")
            continue
        
        # ===== BREAK EVEN LOGIC =====
        if not trade.get("be_active", False):
            if direction == "LONG":
                halfway = entry + ((tp - entry) * 0.5)
                
                if current_price >= halfway:
                    trade["sl"] = entry
                    trade["be_active"] = True
                    modified = True
                    
                    await send_alert(
                        f"🟡 #{trade_no}\n\n"
                        f"{symbol}\n\n"
                        f"BE ACTIVE\n"
                        f"SL → ENTRY"
                    )
            else:  # SHORT
                halfway = entry - ((entry - tp) * 0.5)
                
                if current_price <= halfway:
                    trade["sl"] = entry
                    trade["be_active"] = True
                    modified = True
                    
                    await send_alert(
                        f"🟡 #{trade_no}\n\n"
                        f"{symbol}\n\n"
                        f"BE ACTIVE\n"
                        f"SL → ENTRY"
                    )
        
        # ===== TRAILING ACTIVATION =====
        if not trade.get("trail_active", False):
            if direction == "LONG":
                trigger = entry + ((tp - entry) * 0.75)
                
                if current_price >= trigger:
                    trade["trail_active"] = True
                    modified = True
                    
                    await send_alert(
                        f"🚀 #{trade_no}\n\n"
                        f"{symbol}\n\n"
                        f"TRAILING ACTIVE"
                    )
            else:  # SHORT
                trigger = entry - ((entry - tp) * 0.75)
                
                if current_price <= trigger:
                    trade["trail_active"] = True
                    modified = True
                    
                    await send_alert(
                        f"🚀 #{trade_no}\n\n"
                        f"{symbol}\n\n"
                        f"TRAILING ACTIVE"
                    )
        
        # ===== MOVE TRAILING SL =====
        if trade.get("trail_active", False):
            if direction == "LONG":
                new_sl = current_price * 0.995  # 0.5% trail
                
                if new_sl > trade["sl"]:
                    old_sl = trade["sl"]
                    trade["sl"] = new_sl
                    modified = True
                    logger.info(f"TRAIL UPDATE: {symbol} SL {old_sl:.6f} → {new_sl:.6f}")
            
            else:  # SHORT
                new_sl = current_price * 1.005  # 0.5% trail
                
                if new_sl < trade["sl"]:
                    old_sl = trade["sl"]
                    trade["sl"] = new_sl
                    modified = True
                    logger.info(f"TRAIL UPDATE: {symbol} SL {old_sl:.6f} → {new_sl:.6f}")
        
        # ===== CHECK TP / SL HIT =====
        sl = float(trade["sl"])  # Re-read in case it was updated
        
        if direction == "LONG":
            pnl = (current_price - entry) * qty
            hit_tp = current_price >= tp
            hit_sl = current_price <= sl
        
        else:  # SHORT
            pnl = (entry - current_price) * qty
            hit_tp = current_price <= tp
            hit_sl = current_price >= sl
        
        if hit_tp or hit_sl:
            result = "WIN" if hit_tp else "LOSS"
            
            # Close the trade
            close_trade(symbol, current_price, result)
            update_balance(pnl)
            
            balance = get_balance()["balance"]
            
            # FIXED: Better alert formatting with emoji and details
            status_emoji = "✅" if hit_tp else "❌"
            pnl_sign = "+" if pnl >= 0 else ""
            
            await send_alert(
                f"{status_emoji} #{trade_no}\n\n"
                f"{symbol}\n\n"
                f"{result}\n\n"
                f"Entry  {entry:.6f}\n"
                f"Exit   {current_price:.6f}\n"
                f"Price  {abs(current_price - entry):.6f}\n\n"
                f"PnL: {pnl_sign}${pnl:.2f}\n"
                f"Balance: ${balance:.2f}"
            )
            
            # Mark for removal
            closed_indices.append(idx)
            modified = True
    
    # Save updated trades
    if modified:
        # Filter out closed trades
        updated_trades = [t for i, t in enumerate(trades) if i not in closed_indices]
        save_open_trades(updated_trades)
        logger.info(f"Trades updated. Remaining open: {len(updated_trades)}")
