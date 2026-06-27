import ccxt
import logging
import os

logger = logging.getLogger(__name__)


def get_exchange():
    execute = os.getenv("EXECUTE_TRADES", "false").lower() == "true"

    if not execute:
        logger.info("Paper mode - no real exchange connection")
        return None  # Dummy for paper mode

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
        config["urls"] = {
            'api': {
                'public': 'https://api-testnet.bybit.com',
                'private': 'https://api-testnet.bybit.com'
            }
        }
        logger.info("🚀 Bybit Testnet")
    else:
        logger.info("Bybit Live")

    exchange = ccxt.bybit(config)
    exchange.load_markets()
    logger.info(f"Loaded {len(exchange.markets)} markets")
    return exchange
