import os
import logging
from exchange import get_trade_client

logger = logging.getLogger(__name__)

client = get_trade_client()

EXECUTE = os.getenv("EXECUTE_TRADES", "false").lower() == "true"


async def execute_trade(signal):
    logger.info(f"EXECUTE_TRADES={EXECUTE}")
    logger.info(f"Received signal: {signal}")

    if not EXECUTE:
        logger.warning("Trade execution disabled (paper mode).")
        return {"paper": True}

    try:
        symbol = signal["symbol"].replace("/", "").upper()
        direction = signal["direction"]
        qty = str(signal["qty"])
        sl = signal.get("sl")

        side = "Buy" if direction == "LONG" else "Sell"

        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": qty,
        }

        if sl is not None:
            params["stopLoss"] = str(sl)

        logger.info(f"Sending order to Bybit: {params}")

        result = client.place_order(**params)

        logger.info(f"Bybit response: {result}")
        logger.info(f"✅ EXECUTED {direction} {symbol} Qty={qty}")

        return result

    except Exception as e:
        logger.exception(f"❌ Trade failed: {e}")
        return None


async def close_position(symbol, direction, qty):
    if not EXECUTE:
        logger.info("Paper mode - close skipped.")
        return True

    try:
        symbol = symbol.replace("/", "").upper()
        side = "Sell" if direction == "LONG" else "Buy"

        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "reduceOnly": True,
        }

        logger.info(f"Closing position: {params}")

        result = client.place_order(**params)

        logger.info(f"Close response: {result}")

        return result

    except Exception as e:
        logger.exception(f"❌ Close failed: {e}")
        return None


async def activate_trailing_stop(symbol, direction, qty, trail=0.5):
    if not EXECUTE:
        logger.info("Paper mode - trailing stop skipped.")
        return None

    try:
        symbol = symbol.replace("/", "").upper()

        params = {
            "category": "linear",
            "symbol": symbol,
            "trailingStop": str(trail),
        }

        logger.info(f"Setting trailing stop: {params}")

        result = client.set_trading_stop(**params)

        logger.info(f"Trailing stop response: {result}")

        return result

    except Exception as e:
        logger.exception(f"❌ Trailing stop failed: {e}")
        return None
