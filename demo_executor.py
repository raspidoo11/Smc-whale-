import logging
import os
from exchange import get_exchange
from trade_manager import get_balance

logger = logging.getLogger(__name__)
client = get_exchange()


async def execute_trade(signal):
    if os.getenv("EXECUTE_TRADES", "false").lower() != "true":
        logger.info("Execution disabled.")
        return {"result": {"orderId": "paper"}}

    try:
        symbol = signal["symbol"].replace("/", "")
        direction = signal["direction"]
        qty = signal["qty"]

        side = "Buy" if direction == "LONG" else "Sell"

        order = client.place_active_order(
            symbol=symbol,
            side=side,
            order_type="Market",
            qty=qty,
            time_in_force="GoodTillCancel"
        )

        logger.info(f"✅ EXECUTED {direction} {symbol} | Qty: {qty}")
        return order

    except Exception as e:
        logger.exception(f"Order failed: {e}")
        return None


async def close_position(symbol, direction):
    if os.getenv("EXECUTE_TRADES", "false").lower() != "true":
        return True

    try:
        side = "Sell" if direction == "LONG" else "Buy"
        # Close position logic (simplified)
        order = client.place_active_order(
            symbol=symbol.replace("/", ""),
            side=side,
            order_type="Market",
            qty=0.01,  # Adjust based on position
            time_in_force="GoodTillCancel",
            reduce_only=True
        )
        logger.info(f"Closed {direction} position on {symbol}")
        return order
    except Exception as e:
        logger.error(f"Close failed: {e}")
        return None
