import ccxt
import logging
import os

logger = logging.getLogger(__name__)


def get_exchange():
    execute = os.getenv("EXECUTE_TRADES", "false").lower() == "true"

    config = {
        "enableRateLimit": True,
        "timeout": 30000,
        "options": {
            "defaultType": "swap",
            "defaultSettle": "USDT"
        }
    }

    mode = os.getenv("TRADE_MODE", "demo").lower()

    if execute:
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")

        if api_key and api_secret:
            config["apiKey"] = api_key
            config["secret"] = api_secret

        if mode == "demo":
            # UTA Demo under main account uses production API
            config["options"]["testnet"] = False
            config["options"]["demo"] = True  # UTA demo flag
            logger.info("Bybit UTA Demo Trading (Main Account)")
        else:
            logger.info("Bybit Live Trading")
    else:
        logger.info("Paper mode - public data only")

    exchange = ccxt.bybit(config)
    exchange.load_markets()
    logger.info(f"Loaded {len(exchange.markets)} markets")
    return exchange
