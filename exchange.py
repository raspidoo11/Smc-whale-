import bybit
import logging
import os

logger = logging.getLogger(__name__)


def get_exchange():
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    mode = os.getenv("TRADE_MODE", "demo").lower()

    if mode == "demo":
        client = bybit.bybit(test=True, api_key=api_key, api_secret=api_secret)
        logger.info("🚀 Using Bybit Testnet SDK")
    else:
        client = bybit.bybit(test=False, api_key=api_key, api_secret=api_secret)
        logger.info("Using Bybit Live SDK")

    return client
