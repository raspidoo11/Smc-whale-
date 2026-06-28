import logging
from trade_manager import get_balance

logger = logging.getLogger(__name__)


def calculate_qty(entry, sl):
    """Futures-style sizing with 10x leverage"""

    balance = get_balance()["balance"]
    risk_usd = balance * 0.05   # 5% risk

    stop_distance_pct = abs(
        entry - sl
    ) / entry

    if stop_distance_pct <= 0:
        return 0

    position_value = (
        risk_usd /
        stop_distance_pct
    )

    max_position = (
        balance *
        10   # 10x leverage
    )

    position_value = min(
        position_value,
        max_position
    )

    qty = (
        position_value /
        entry
    )

    return round(
        qty,
        6
    )
