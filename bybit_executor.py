import logging
import time
from exchange import get_trade_client
from trade_manager import get_risk_amount
from xgboost_trainer import get_ai_risk_percent, detect_market_regime

logger = logging.getLogger(__name__)


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


async def activate_trailing_stop(symbol, direction, qty, trail_percent=0.5, active_price=None):
    client = get_trade_client()
    try:
        # FIX: execute_trade() normalizes ccxt-style symbols
        # (e.g. "RAVE/USDT:USDT" -> "RAVEUSDT") before sending to Bybit's
        # raw API, but this function was taking `symbol` as-is. Whatever
        # calls this (trade_monitor.py) was passing the ccxt-style symbol
        # straight from the trade record, which Bybit's API rejects with
        # ErrCode 10001 ("symbol invalid"). Normalizing here makes this
        # safe regardless of what format the caller passes in.
        symbol = symbol.split(":")[0].replace("/", "").upper()

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
        logger.info(f"🚀 Trailing stop activated: {symbol} | {trail_percent}%")
        return result

    except Exception as e:
        logger.exception(f"Trailing stop failed: {e}")
        return None
