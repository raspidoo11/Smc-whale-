import ccxt
import logging
import os

logger = logging.getLogger(__name__)


def get_exchange():
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    mode = os.getenv("TRADE_MODE", "demo").lower()

    config = {
        "enableRateLimit": True,
        "timeout": 30000,
        "options": {
            "defaultType": "swap",
            "defaultSettle": "USDT",
            "recvWindow": 10000
        }
    }

    if api_key and api_secret:
        config["apiKey"] = api_key
        config["secret"] = api_secret

    if mode == "demo":
        config["options"]["testnet"] = True
        # Explicit Testnet URLs
        config["urls"] = {
            'api': {
                'public': 'https://api-testnet.bybit.com',
                'private': 'https://api-testnet.bybit.com'
            }
        }
        logger.info("🚀 Using Bybit Testnet (https://api-testnet.bybit.com)")
    else:
        logger.info("Using Bybit Live")

    exchange = ccxt.bybit(config)
    exchange.load_markets()

    logger.info(f"Loaded {len(exchange.markets)} markets")
    return exchange
