import os
import logging
from exchange import get_trade_client

client = get_trade_client()

logger = logging.getLogger(__name__)

client = get_exchange()

EXECUTE = os.getenv("EXECUTE_TRADES", "false").lower() == "true"


async def execute_trade(signal):
    if not EXECUTE:
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

        if sl:
            params["stopLoss"] = str(sl)

        result = client.place_order(**params)

        logger.info(f"EXECUTED {direction} {symbol} qty={qty}")
        return result

    except Exception as e:
        logger.exception(f"Trade failed: {e}")
        return None


async def close_position(symbol, direction, qty):
    try:
        symbol = symbol.replace("/", "").upper()
        side = "Sell" if direction == "LONG" else "Buy"

        return client.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
            reduceOnly=True,
        )

    except Exception as e:
        logger.error(f"Close failed: {e}")
        return None


async def activate_trailing_stop(symbol, direction, qty, trail=0.5):
    try:
        symbol = symbol.replace("/", "").upper()
        side = "Sell" if direction == "LONG" else "Buy"

        # ✔ V5 trailing stop is position-based (NOT order-based)
        result = client.set_trading_stop(
            category="linear",
            symbol=symbol,
            trailingStop=str(trail),
        )

        logger.info(f"Trailing stop set on {symbol} = {trail}%")
        return result

    except Exception as e:
        logger.exception(f"Trailing stop failed: {e}")
        return None
