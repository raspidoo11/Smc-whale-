import logging
from trade_manager import risk_amount

logger = logging.getLogger(__name__)


def calculate_qty(entry, sl):
    """Calculate position size based on risk"""
    risk = risk_amount()
    distance = abs(entry - sl)

    if distance <= 0:
        return 0.0

    qty = risk / distance
    return round(qty, 6)
