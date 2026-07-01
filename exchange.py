import os
import logging
import ccxt
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)

_public_exchange = None
_trade_client = None


def get_exchange():
    """
    CCXT exchange for public market data only.
    """
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
        logger.info(f"✅ Loaded {_public_exchange.id} markets")

    return _public_exchange


def get_trade_client():
    """
    Pybit client for authenticated trading.
    """
    global _trade_client

    if _trade_client is not None:
        return _trade_client

    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")

    if not api_key or not api_secret:
        raise ValueError("Missing BYBIT_API_KEY or BYBIT_API_SECRET")

    mode = os.getenv("TRADE_MODE", "demo").lower()

    if mode == "demo":
        logger.info("🎯 Connecting to Bybit Demo Trading...")
        _trade_client = HTTP(
            testnet=False,
            demo=True,
            api_key=api_key,
            api_secret=api_secret,
        )

    elif mode == "testnet":
        logger.info("🧪 Connecting to Bybit Testnet...")
        _trade_client = HTTP(
            testnet=True,
            demo=False,
            api_key=api_key,
            api_secret=api_secret,
        )

    elif mode == "live":
        logger.info("💰 Connecting to Bybit Live...")
        _trade_client = HTTP(
            testnet=False,
            demo=False,
            api_key=api_key,
            api_secret=api_secret,
        )

    else:
        raise ValueError(f"Unknown TRADE_MODE: {mode}")

    try:
        response = _trade_client.get_wallet_balance(accountType="UNIFIED")

        if response.get("retCode") == 0:
            logger.info("✅ Successfully connected to Bybit API")
            logger.info(f"🌐 Trading Mode: {mode.upper()}")

            wallets = response.get("result", {}).get("list", [])
            logger.info(f"💰 Wallets detected: {len(wallets)}")

        else:
            logger.error(f"❌ Bybit authentication failed: {response.get('retMsg')}")

    except Exception as e:
        logger.exception(f"❌ Could not connect to Bybit: {e}")
        raise

    logger.info(f"🚀 Trading client initialized ({mode})")

    return _trade_client
