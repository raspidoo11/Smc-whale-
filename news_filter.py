import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# High-impact economic events (you can expand this list or load from a calendar)
HIGH_IMPACT_EVENTS = [
    # Format: (weekday, hour_utc, event_name)
    (1, 13, "FOMC Interest Rate Decision"),      # Tuesday ~13:00 UTC
    (3, 12, "CPI Release"),                       # Thursday ~12:00 UTC
    (4, 12, "Non-Farm Payrolls (NFP)"),           # Friday ~12:30 UTC
    (2, 14, "Fed Chair Speech"),                  # Wednesday ~14:00 UTC
]


def is_high_impact_news_time(minutes_before=30, minutes_after=30):
    """
    Check if we are currently in a high-impact news window.
    Returns True if we should avoid trading.
    """
    now = datetime.now(timezone.utc)
    current_weekday = now.weekday()       # 0=Monday ... 6=Sunday
    current_hour = now.hour
    current_minute = now.minute

    for event_weekday, event_hour, event_name in HIGH_IMPACT_EVENTS:
        if current_weekday != event_weekday:
            continue

        event_time = now.replace(hour=event_hour, minute=0, second=0, microsecond=0)

        # Check if we are within the danger window
        time_diff = (now - event_time).total_seconds() / 60

        if -minutes_before <= time_diff <= minutes_after:
            logger.warning(f"🚫 High-impact news window active: {event_name}")
            return True

    return False


def get_news_status():
    """Returns whether we should pause trading due to news"""
    if is_high_impact_news_time():
        return True, "High-impact news window active"
    return False, "No major news"
