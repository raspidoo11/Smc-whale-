import logging
from trade_manager import risk_amount  # We'll add this to trade_manager if needed

logger = logging.getLogger(__name__)


def calculate_qty(entry, sl):
    """Calculate position size based on risk"""
    risk = 100.0 * 0.01  # Default 1% risk if no balance loaded
    try:
        # Try to use advanced risk from trade_manager
        risk = risk_amount() if 'risk_amount' in globals() else risk
    except:
        pass

    distance = abs(entry - sl)
    if distance <= 0:
        return 0.0

    qty = risk / distance
    return round(qty, 6)
