import os
import logging
import ccxt
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)

_public_exchange = None
_trade_client = None


def get_exchange():
    """CCXT exchange for public market data."""
    global _public_exchange

    if _public_exchange is None:
        _public_exchange = ccxt.bybit({
            "enableRateLimit": True,
            "timeout": 30000,
            "options": {
                "defaultType": "swap",
                "defaultSettle": "USDT",
            },
        })

        _public_exchange.load_markets()
        logger.info(f"Loaded {_public_exchange.id} markets")

    return _public_exchange


def get_trade_client():
    """Pybit client for authenticated trading."""
    global _trade_client

    if _trade_client is not None:
        return _trade_client

    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")

    if not api_key or not api_secret:
        raise ValueError("Missing Bybit API credentials.")

    mode = os.getenv("TRADE_MODE", "testnet").lower()

    _trade_client = HTTP(
        testnet=(mode == "testnet"),
        api_key=api_key,
        api_secret=api_secret,
    )

    logger.info(f"Trading client initialized ({mode})")

    return _trade_client
