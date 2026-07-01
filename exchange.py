import os
import logging
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)


def get_exchange():
    execute = os.getenv("EXECUTE_TRADES", "false").lower() == "true"
    mode = os.getenv("TRADE_MODE", "testnet").lower()

    if not execute:
        logger.info("Paper mode - no authentication")
        return None

    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")

    if not api_key or not api_secret:
        raise ValueError("Missing BYBIT_API_KEY or BYBIT_API_SECRET")

    # ✔ BYBIT V5 SESSION
    session = HTTP(
        testnet=(mode == "testnet"),
        api_key=api_key,
        api_secret=api_secret,
    )

    # quick validation call (fails fast if keys are wrong)
    try:
        info = session.get_wallet_balance(accountType="UNIFIED")
        logger.info("Bybit auth successful ✔")
    except Exception as e:
        logger.error(f"Bybit auth failed ❌: {e}")
        raise

    return session
