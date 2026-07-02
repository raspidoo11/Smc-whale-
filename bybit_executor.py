import logging
import os
import time
from exchange import get_trade_client
from trade_manager import get_risk_amount
from xgboost_trainer import get_ai_risk_percent, detect_market_regime

logger = logging.getLogger(__name__)

# Confirm this exact variable name matches what you set in Railway. If it
# doesn't exist in the environment at all, this defaults to True (trades
# execute) -- so double check the Railway Variables tab shows exactly
# EXECUTE_TRADES = false, not a typo'd name.
EXECUTE_TRADES = os.getenv("EXECUTE_TRADES", "true").lower() == "true"


def get_symbol_info(symbol):
    client = get_trade_client()
    try:
        markets = client.load_markets()
        clean_symbol = symbol.replace("/", "").upper()
        if clean_symbol in markets:
            return markets[clean_symbol]
        return None
    except Exception as e:
        logger.error(f"Failed to load market info for {symbol}: {e}")
        return None


def calculate_proper_qty(symbol, entry_price, sl_price, ai_prob=50, regime="ranging", recent_drawdown=0.0):
    """
    Calculate quantity using AI-driven dynamic risk sizing.
    """
    market = get_symbol_info(symbol)
    if not market:
        logger.error(f"Could not get market info for {symbol}")
        return None

    try:
        risk_per_unit = abs(entry_price - sl_price)
        if risk_per_unit == 0:
            return None

        # Get base risk amount (0.5% of balance)
        base_risk = get_risk_amount()

        # Adjust risk using AI confidence
        adjusted_risk = base_risk * (get_ai_risk_percent(ai_prob, recent_drawdown, regime) / 0.5)

        raw_qty = adjusted_risk / risk_per_unit

        # Apply exchange precision
        step = market.get("precision", {}).get("amount", 0.001)
        qty = round(raw_qty / step) * step

        # Check limits
        min_qty = market.get("limits", {}).get("amount", {}).get("min", 0)
        max_qty = market.get("limits", {}).get("amount", {}).get("max", 999999)

        if qty < min_qty:
            logger.warning(f"Quantity {qty} below minimum for {symbol}")
            return None
        if qty > max_qty:
            qty = max_qty

        return round(qty, 6)

    except Exception as e:
        logger.exception(f"Quantity calculation failed for {symbol}: {e}")
        return None


def set_leverage_if_needed(symbol, desired_leverage=10):
    client = get_trade_client()
    try:
        positions = client.fetch_positions([symbol])
        current_lev = int(positions[0].get("leverage", 0)) if positions else 0

        if current_lev != desired_leverage:
            logger.info(f"Setting leverage to {desired_leverage}x for {symbol}")
            client.set_leverage(desired_leverage, symbol, params={"category": "linear"})
            time.sleep(0.5)
        return True
    except Exception as e:
        logger.error(f"Leverage setting failed for {symbol}: {e}")
        return False


async def execute_trade(signal):
    if not EXECUTE_TRADES:
        logger.info(f"⏸️ EXECUTE_TRADES=false — skipping order for {signal.get('symbol')} (no Bybit call made)")
        return None

    client = get_trade_client()

    try:
        symbol = signal["symbol"].split(":")[0].replace("/", "").upper()
        direction = signal["direction"]
        entry = float(signal["entry"])
        sl = float(signal.get("sl", 0))
        tp = float(signal.get("tp", 0))
        ai_prob = float(signal.get("ai_prob", 50))
        regime = signal.get("market_regime", "ranging")

        # Calculate quantity using AI risk sizing
        qty = calculate_proper_qty(symbol, entry, sl, ai_prob=ai_prob, regime=regime)
        if not qty:
            logger.error(f"❌ Invalid quantity for {symbol}")
            return None

        # Set leverage
        if not set_leverage_if_needed(symbol):
            logger.error(f"❌ Could not set leverage for {symbol}")
            return None

        side = "Buy" if direction == "LONG" else "Sell"

        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
        }

        if sl:
            params["stopLoss"] = str(sl)
        if tp:
            params["takeProfit"] = str(tp)

        logger.info(f"📤 Sending order: {params}")

        result = None
        for attempt in range(3):
            try:
                result = client.place_order(**params)
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                logger.warning(f"Retrying order ({attempt + 1}/3)")
                time.sleep(1.5)

        logger.info(f"📥 Bybit response: {result}")

        if result and result.get("retCode") == 0:
            logger.info(f"✅ Order placed: {symbol} | Qty: {qty} | AI Risk Adjusted")

        return result

    except Exception as e:
        logger.exception(f"❌ Trade execution failed: {e}")
        return None


def get_open_position_size(symbol):
    """Ask the exchange directly whether this position is still open, and if
    so, how large. Local trade records can go stale if Bybit auto-closes a
    position via its own SL/TP before our monitor loop notices."""
    client = get_trade_client()
    try:
        positions = client.fetch_positions([symbol])
        for pos in positions:
            size = float(pos.get("contracts") or pos.get("size") or 0)
            if size > 0:
                return size
        return 0.0
    except Exception as e:
        logger.error(f"Failed to fetch live position size for {symbol}: {e}")
        return None  # None = "couldn't check", distinct from 0 = "confirmed closed"


def get_last_closed_pnl(symbol):
    """
    Once a trailing stop has actually closed a position, pull the real exit
    price and realized PnL from Bybit's closed-PnL record instead of
    guessing from the last polled ticker price. Best-effort: if this fails
    (API surface differs, permissions, etc.) callers should fall back to an
    approximate price rather than blocking the trade from closing locally.
    """
    client = get_trade_client()
    try:
        symbol = symbol.split(":")[0].replace("/", "").upper()
        resp = client.get_closed_pnl(category="linear", symbol=symbol, limit=1)
        rows = resp.get("result", {}).get("list", [])
        if not rows:
            return None
        row = rows[0]
        exit_price = float(row.get("avgExitPrice", 0) or 0)
        return {
            "exit_price": exit_price if exit_price > 0 else None,
            "realized_pnl": float(row.get("closedPnl", 0) or 0),
            "qty": float(row.get("qty", 0) or 0),
        }
    except Exception as e:
        logger.debug(f"Could not fetch closed PnL for {symbol}: {e}")
        return None


async def activate_trailing_stop(symbol, direction, qty, trail_percent=0.5, active_price=None):
    if not EXECUTE_TRADES:
        logger.info(f"⏸️ EXECUTE_TRADES=false — skipping trailing stop for {symbol} (no Bybit call made)")
        return None

    client = get_trade_client()
    try:
        symbol = symbol.split(":")[0].replace("/", "").upper()

        # FIX: Bybit closes positions automatically via the stopLoss/
        # takeProfit attached at entry, which can happen before our local
        # monitor notices. Check the real position size first instead of
        # trusting the stored trade record's qty -- this is what was
        # causing repeated "Qty invalid" errors on trades that had already
        # closed exchange-side.
        live_size = get_open_position_size(symbol)

        if live_size is None:
            logger.warning(f"Could not verify position size for {symbol}, skipping trailing stop attempt")
            return None

        if live_size == 0:
            logger.info(f"Position for {symbol} already closed exchange-side, skipping trailing stop")
            return None

        # Round to the symbol's actual precision instead of trusting the
        # qty passed in from the local trade record, which may have been
        # computed with different rounding than what Bybit expects now.
        market = get_symbol_info(symbol)
        step = market.get("precision", {}).get("amount", 0.001) if market else 0.001
        adjusted_qty = round(live_size / step) * step
        adjusted_qty = round(min(adjusted_qty, live_size), 6)

        side = "Sell" if direction == "LONG" else "Buy"

        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(adjusted_qty),
            "reduceOnly": True,
            "trailingStop": str(trail_percent),
        }

        if active_price:
            params["activePrice"] = str(active_price)

        result = client.place_order(**params)
        logger.info(f"🚀 Trailing stop activated: {symbol} | {trail_percent}% | qty={adjusted_qty}")
        return result

    except Exception as e:
        logger.exception(f"Trailing stop failed: {e}")
        return None
