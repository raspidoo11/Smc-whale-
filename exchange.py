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
        # Explicit Testnet URLs as per Bybit docs
        config["urls"] = {
            'api': {
                'public': 'https://api-testnet.bybit.com',
                'private': 'https://api-testnet.bybit.com'
            }
        }
        logger.info("🚀 Bybit Testnet (https://api-testnet.bybit.com)")
    else:
        logger.info("Bybit Live")

    exchange = ccxt.bybit(config)

    try:
        exchange.load_markets()
        logger.info(f"✅ Loaded {len(exchange.markets)} markets")
    except Exception as e:
        logger.error(f"Failed to load markets: {e}")

    return exchange
