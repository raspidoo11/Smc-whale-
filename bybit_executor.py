import logging
import time
from exchange import get_trade_client

logger = logging.getLogger(__name__)


def get_symbol_info(symbol):
    """Get market info for precision and limits"""
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


def calculate_proper_qty(symbol, entry_price, sl_price, risk_usd=5.0):
    """
    Calculate quantity with proper exchange precision and limits.
    Risks approximately $risk_usd when SL is hit.
    """
    market = get_symbol_info(symbol)
    if not market:
        logger.error(f"Could not get market info for {symbol}")
        return None

    try:
        risk_per_unit = abs(entry_price - sl_price)
        if risk_per_unit == 0:
            return None

        raw_qty = risk_usd / risk_per_unit

        # Apply exchange precision
        qty = market["precision"]["amount"] and float(
            market["precision"]["amount"]
        ) or 0.001

        # Round to correct step
        qty = round(raw_qty / qty) * qty

        # Check min/max
        min_qty = market.get("limits", {}).get("amount", {}).get("min", 0)
        max_qty = market.get("limits", {}).get("amount", {}).get("max", 999999)

        if qty < min_qty:
            logger.warning(f"Quantity {qty} below minimum {min_qty} for {symbol}")
            return None
        if qty > max_qty:
            qty = max_qty

        return round(qty, 6)

    except Exception as e:
        logger.exception(f"Quantity calculation failed for {symbol}: {e}")
        return None


def set_leverage_if_needed(symbol, desired_leverage=10):
    """Ensure correct leverage is set before trading"""
    client = get_trade_client()
    try:
        # Check current leverage
        positions = client.fetch_positions([symbol])
        current_lev = None
        if positions:
            current_lev = int(positions[0].get("leverage", 0))

        if current_lev != desired_leverage:
            logger.info(f"Setting leverage to {desired_leverage}x for {symbol}")
            client.set_leverage(desired_leverage, symbol, params={"category": "linear"})
            time.sleep(0.5)  # small delay after leverage change
        return True
    except Exception as e:
        logger.error(f"Leverage setting failed for {symbol}: {e}")
        return False


async def execute_trade(signal):
    """Execute trade with proper quantity, leverage, and SL/TP"""
    client = get_trade_client()

    try:
        symbol = signal["symbol"].split(":")[0].replace("/", "").upper()
        direction = signal["direction"]
        entry = float(signal["entry"])
        sl = float(signal.get("sl", 0))
        tp = float(signal.get("tp", 0))

        # 1. Calculate proper quantity
        qty = calculate_proper_qty(symbol, entry, sl)
        if not qty:
            logger.error(f"❌ Invalid quantity calculated for {symbol}")
            return None

        # 2. Ensure correct leverage
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

        # 3. Place order with retry for temporary errors
        result = None
        for attempt in range(3):
            try:
                result = client.place_order(**params)
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                logger.warning(f"Retrying order ({attempt + 1}/3) due to: {e}")
                time.sleep(1.5)

        logger.info(f"📥 Bybit response: {result}")

        # 4. Verify SL/TP were attached (basic check)
        if result and result.get("retCode") == 0:
            logger.info(f"✅ Order placed successfully for {symbol}")
            # You can add position verification here later using fetch_positions
        else:
            logger.warning(f"⚠️ Order may not have attached SL/TP properly")

        return result

    except Exception as e:
        logger.exception(f"❌ Trade execution failed: {e}")
        return None


async def activate_trailing_stop(symbol, direction, qty, trail_percent=0.5, active_price=None):
    """
    Activate Bybit native trailing stop.
    Call this after TP is reached or manually.
    """
    client = get_trade_client()
    try:
        side = "Sell" if direction == "LONG" else "Buy"

        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "reduceOnly": True,
            "trailingStop": str(trail_percent),
        }

        if active_price:
            params["activePrice"] = str(active_price)

        result = client.place_order(**params)
        logger.info(f"🚀 Trailing stop activated on {symbol} | {trail_percent}%")
        return result

    except Exception as e:
        logger.exception(f"Trailing stop activation failed: {e}")
        return None
