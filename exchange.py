import ccxt
import logging

logger = logging.getLogger(__name__)


def get_exchange():
    try:
        logger.info(
            "Connecting to Bybit..."
        )

        exchange = ccxt.bybit({
            "enableRateLimit": True,
            "timeout": 30000,
            "options": {
                "defaultType": "swap",
                "defaultSettle": "USDT"
            }
        })

        exchange.load_markets()

        logger.info(
            f"Loaded {len(exchange.markets)} markets"
        )

        return exchange

    except Exception as e:
        logger.exception(
            f"Exchange startup failed: {e}"
        )
        raise
