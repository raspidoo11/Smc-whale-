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

        # Get current balance
        balance_data = get_balance()
        total_balance = balance_data.get("balance", 100.0)

        # Risk 5% of balance
        risk_amount = total_balance * 0.05

        # Distance to SL
        distance = abs(entry - sl)
        if distance <= 0:
            distance = entry * 0.01  # fallback 1%

        # Calculate base quantity (with 10x leverage)
        qty = (risk_amount / distance) * 10  # 10x leverage

        # Round to reasonable size
        qty = round(qty, 6)

        side = "buy" if direction == "LONG" else "sell"

        order = exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=qty,
            params={"leverage": 10}  # Set 10x leverage
        )

        logger.info(f"✅ EXECUTED {direction} {symbol} | Qty: {qty} | 5% risk | 10x lev")
        return order

    except Exception as e:
        logger.exception(f"Order failed: {e}")
        return None
