import logging
from exchange import get_exchange
from trade_manager import get_balance

logger = logging.getLogger(__name__)
exchange = get_exchange()


async def execute_trade(signal):
    """Place market order using max 5% of balance + 10x leverage"""
    try:
        symbol = signal["symbol"]
        direction = signal["direction"]
        entry = signal["entry"]
        sl = signal["sl"]

        balance_data = get_balance()
        total_balance = balance_data.get("balance", 100.0)

        risk_amount = total_balance * 0.05
        distance = abs(entry - sl) or (entry * 0.01)

        qty = (risk_amount / distance) * 10   # 10x leverage
        qty = round(qty, 6)

        side = "buy" if direction == "LONG" else "sell"

        order = exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=qty,
            params={"leverage": 10}
        )

        logger.info(f"✅ EXECUTED {direction} {symbol} | Qty: {qty} | 5% risk")
        return order

    except Exception as e:
        logger.exception(f"Order failed: {e}")
        return None


async def close_position(symbol, direction):
    """Close full position"""
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
