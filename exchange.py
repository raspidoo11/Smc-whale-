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
            "defaultSettle": "USDT"
        }
    }

    if api_key and api_secret:
        config["apiKey"] = api_key
        config["secret"] = api_secret

    if mode == "demo":
        config["options"]["testnet"] = True
        logger.info("🚀 Connected to Bybit DEMO (Testnet)")
    else:
        logger.info("⚠️ Connected to Bybit LIVE")

    exchange = ccxt.bybit(config)

    # Important: Load markets after setting credentials
    exchange.load_markets()

    logger.info(f"Loaded {len(exchange.markets)} markets")
    return exchange
