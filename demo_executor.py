import logging
import os
from exchange import get_exchange
from trade_manager import get_balance

logger = logging.getLogger(__name__)
exchange = get_exchange()


async def test_connection():
    """Test Bybit API connection"""
    try:
        balance = exchange.fetch_balance()
        logger.info(f"✅ API Connection OK | Balance: {balance.get('total', {}).get('USDT', 'N/A')} USDT")
        return True
    except Exception as e:
        logger.error(f"❌ API Connection FAILED: {e}")
        return False


async def execute_trade(signal):
    if os.getenv("EXECUTE_TRADES", "false").lower() != "true":
        logger.info("Execution disabled. Would have executed trade.")
        return {"id": "paper_order"}

    try:
        symbol = signal["symbol"]
        direction = signal["direction"]
        entry = signal["entry"]
        sl = signal["sl"]

        balance_data = get_balance()
        total_balance = balance_data.get("balance", 100.0)

        risk_amount = total_balance * 0.05
        distance = abs(entry - sl) or (entry * 0.01)

        qty = (risk_amount / distance) * 10
        qty = round(qty, 6)

        side = "buy" if direction == "LONG" else "sell"

        order = exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=qty,
            params={"leverage": 10}
        )

        logger.info(f"✅ EXECUTED {direction} {symbol} | Qty: {qty}")
        return order

    except Exception as e:
        logger.exception(f"Order failed: {e}")
        return None


async def close_position(symbol, direction):
    if os.getenv("EXECUTE_TRADES", "false").lower() != "true":
        logger.info("Execution disabled. Would have closed position.")
        return True

    try:
        side = "sell" if direction == "LONG" else "buy"
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if pos['symbol'] == symbol and float(pos.get('contracts', 0)) > 0:
                amount = float(pos['contracts'])
                order = exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=side,
                    amount=amount
                )
                logger.info(f"Closed {direction} position on {symbol}")
                return order
        return None
    except Exception as e:
        logger.error(f"Close failed for {symbol}: {e}")
        return None
