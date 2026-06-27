import logging
from exchange import get_exchange
from trade_manager import risk_amount

logger = logging.getLogger(__name__)
exchange = get_exchange()


async def execute_trade(signal):
    """Place market order on Bybit Demo"""
    try:
        symbol = signal["symbol"]
        direction = signal["direction"]
        qty = signal["qty"]

        side = "buy" if direction == "LONG" else "sell"

        order = exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=qty
        )

        logger.info(f"✅ EXECUTED {direction} {symbol} | Qty: {qty}")
        return order

    except Exception as e:
        logger.exception(f"Order failed: {e}")
        return None


async def close_position(symbol, side):
    """Close position (market order opposite side)"""
    try:
        position_side = "sell" if side == "buy" else "buy"
        # Get current position size
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if pos['symbol'] == symbol and float(pos['contracts']) > 0:
                amount = float(pos['contracts'])
                order = exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=position_side,
                    amount=amount
                )
                logger.info(f"Closed position on {symbol}")
                return order
        return None
    except Exception as e:
        logger.error(f"Close failed for {symbol}: {e}")
        return None
