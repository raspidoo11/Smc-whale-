import logging
from exchange import get_trade_client

logger = logging.getLogger(__name__)


async def execute_trade(signal):
    try:
        client = get_trade_client()

        symbol = signal["symbol"].split(":")[0].replace("/", "").upper()
        direction = signal["direction"]
        qty = str(signal["qty"])
        sl = signal.get("sl")
        tp = signal.get("tp")

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

        if tp:
            params["takeProfit"] = str(tp)

        logger.info(f"📤 Sending order: {params}")

        result = client.place_order(**params)

        logger.info(f"📥 Bybit response: {result}")

        return result

    except Exception as e:
        logger.exception(f"❌ Trade execution failed: {e}")
        return None
