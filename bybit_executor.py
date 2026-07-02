import logging
import os
import time
from exchange import get_trade_client, get_exchange
from trade_manager import get_risk_amount
from xgboost_trainer import get_ai_risk_percent, detect_market_regime

logger = logging.getLogger(__name__)

# ==========================================================
# CCXT = Market metadata, precision, filters (read-only)
# Pybit  = All trading actions (place_order, leverage, etc.)
# Never mix the two clients.
# ==========================================================


def get_symbol_info(symbol):
    """
    Uses CCXT exchange for market metadata.
    Do NOT call this on the Pybit client.
    """
    exchange = get_exchange()
    try:
        exchange.load_markets()
        # Normalize to CCXT format if needed (e.g. RAVE/USDT:USDT)
        ccxt_symbol = symbol
        if "/" not in ccxt_symbol and ":" not in ccxt_symbol:
            # crude fallback; better if caller passes proper CCXT symbol
            ccxt_symbol = f"{symbol}/USDT:USDT"

        if ccxt_symbol in exchange.markets:
            return exchange.markets[ccxt_symbol]
        # Try without :USDT suffix
        alt = ccxt_symbol.split(":")[0]
        if alt in exchange.markets:
            return exchange.markets[alt]
        return None
    except Exception as e:
        logger.error(f"Failed to load market info for {symbol}: {e}")
        return None


def calculate_proper_qty(symbol, entry_price, sl_price, ai_prob=50, regime="ranging", recent_drawdown=0.0):
    """
    Calculate quantity using AI-driven dynamic risk sizing + CCXT precision.
    """
    market = get_symbol_info(symbol)
    if not market:
        logger.error(f"Could not get market info for {symbol}")
        return None

    try:
        risk_per_unit = abs(entry_price - sl_price)
        if risk_per_unit == 0:
            return None

        base_risk = get_risk_amount()
        adjusted_risk = base_risk * (get_ai_risk_percent(ai_prob, recent_drawdown, regime) / 0.5)

        raw_qty = adjusted_risk / risk_per_unit

        exchange = get_exchange()
        # Use CCXT's proper precision handling (respects Bybit lot sizes)
        qty = float(
            exchange.amount_to_precision(
                market.get("symbol", symbol),
                raw_qty
            )
        )

        # Enforce exchange limits
        min_qty = market.get("limits", {}).get("amount", {}).get("min", 0)
        max_qty = market.get("limits", {}).get("amount", {}).get("max", 999999)

        if qty < min_qty:
            logger.warning(f"Quantity {qty} below minimum for {symbol}")
            return None
        if qty > max_qty:
            qty = max_qty

        return round(qty, 8)

    except Exception as e:
        logger.exception(f"Quantity calculation failed for {symbol}: {e}")
        return None


def set_leverage_if_needed(symbol, desired_leverage=10):
    """
    Uses ONLY Pybit HTTP client.
    Treats "leverage not modified" (110043) as success.
    """
    client = get_trade_client()
    try:
        # Pybit V5 expects string values for leverage
        result = client.set_leverage(
            category="linear",
            symbol=symbol,
            buyLeverage=str(desired_leverage),
            sellLeverage=str(desired_leverage)
        )

        ret_code = result.get("retCode", -1) if isinstance(result, dict) else -1

        if ret_code == 0:
            logger.info(f"Leverage set to {desired_leverage}x for {symbol}")
            return True
        elif ret_code == 110043:
            # "leverage not modified" — treat as success
            logger.info(f"Leverage already {desired_leverage}x for {symbol} (no change needed)")
            return True
        else:
            logger.warning(f"set_leverage returned retCode={ret_code} for {symbol}")
            return False

    except Exception as e:
        # Some versions of pybit raise on 110043 instead of returning it
        if "110043" in str(e):
            logger.info(f"Leverage already {desired_leverage}x for {symbol}")
            return True
        logger.error(f"Leverage setting failed for {symbol}: {e}")
        return False


async def execute_trade(signal):
    """
    Main trade execution entry point.
    Respects EXECUTE_TRADES env var for paper mode.
    """
    # FIX #4: Early exit before any Bybit connection when in paper mode
    if os.getenv("EXECUTE_TRADES", "false").lower() != "true":
        logger.info("📄 Paper trading mode - skipping live order execution")
        return {"paper_mode": True, "signal": signal}

    client = get_trade_client()

    try:
        # Normalize symbol for Bybit (raw format)
        raw_symbol = signal["symbol"].split(":")[0].replace("/", "").upper()
        direction = signal["direction"]
        entry = float(signal["entry"])
        sl = float(signal.get("sl", 0))
        tp = float(signal.get("tp", 0))
        ai_prob = float(signal.get("ai_prob", 50))
        regime = signal.get("market_regime", "ranging")

        # Calculate quantity (uses CCXT for precision)
        qty = calculate_proper_qty(raw_symbol, entry, sl, ai_prob=ai_prob, regime=regime)
        if not qty:
            logger.error(f"❌ Invalid quantity for {raw_symbol}")
            return None

        # Set leverage (pure Pybit)
        if not set_leverage_if_needed(raw_symbol):
            logger.error(f"❌ Could not set leverage for {raw_symbol}")
            return None

        side = "Buy" if direction == "LONG" else "Sell"

        params = {
            "category": "linear",
            "symbol": raw_symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
        }

        if sl:
            params["stopLoss"] = str(sl)
        if tp:
            params["takeProfit"] = str(tp)

        logger.info(f"📤 Sending order: {params}")

        result = None
        for attempt in range(3):
            try:
                result = client.place_order(**params)
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                logger.warning(f"Retrying order ({attempt + 1}/3)")
                time.sleep(1.5)

        logger.info(f"📥 Bybit response: {result}")

        if result and result.get("retCode") == 0:
            logger.info(f"✅ Order placed: {raw_symbol} | Qty: {qty} | AI Risk Adjusted")

        return result

    except Exception as e:
        logger.exception(f"❌ Trade execution failed: {e}")
        return None


async def activate_trailing_stop(symbol, direction, qty, trail_percent=0.5, active_price=None):
    """
    Activates trailing stop using Pybit.
    Respects EXECUTE_TRADES=false (paper mode).
    """
    # Respect paper trading mode - do not place any orders
    if os.getenv("EXECUTE_TRADES", "false").lower() != "true":
        logger.info(f"📄 Paper mode - Skipping trailing stop activation for {symbol}")
        return {"paper_mode": True, "symbol": symbol}

    client = get_trade_client()
    try:
        # Normalize to Bybit raw symbol (important!)
        raw_symbol = symbol.split(":")[0].replace("/", "").upper()

        # Basic qty validation
        if not qty or float(qty) <= 0:
            logger.error(f"Invalid qty for trailing stop on {raw_symbol}: {qty}")
            return None

        side = "Sell" if direction == "LONG" else "Buy"

        params = {
            "category": "linear",
            "symbol": raw_symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "reduceOnly": True,
            "trailingStop": str(trail_percent),
        }

        if active_price:
            params["activePrice"] = str(active_price)

        result = client.place_order(**params)
        logger.info(f"🚀 Trailing stop activated: {raw_symbol} | {trail_percent}%")
        return result

    except Exception as e:
        logger.exception(f"Trailing stop failed: {e}")
        return None
