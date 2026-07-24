import logging
import os
import time
import uuid
from exchange import get_trade_client, get_exchange
from trade_manager import get_risk_amount
from xgboost_trainer import get_ai_risk_percent

logger = logging.getLogger(__name__)

# Confirm this exact variable name matches what you set in Railway. If it
# doesn't exist in the environment at all, this defaults to True (trades
# execute) -- so double check the Railway Variables tab shows exactly
# EXECUTE_TRADES = false, not a typo'd name.
EXECUTE_TRADES = os.getenv("EXECUTE_TRADES", "true").lower() == "true"

CATEGORY = "linear"


def _pybit_symbol(symbol):
    """Normalize any symbol form to Bybit's ('BTCUSDT').
    Accepts ccxt unified ('BTC/USDT:USDT') or already-clean ('BTCUSDT')."""
    return symbol.split(":")[0].replace("/", "").replace("-", "").upper()


def get_symbol_info(symbol):
    """Return the ccxt market dict (precision + limits) for a symbol.

    FIX: this previously called `get_trade_client().load_markets()`, but the
    trade client is a *pybit* HTTP session, which has no `load_markets()` (that
    is a ccxt method). It raised AttributeError on every call, so quantity
    could never be computed and every live/demo order was skipped. Market
    metadata now comes from the ccxt *public* exchange, which is the correct
    source for precision/limits and is already loaded elsewhere.
    """
    ex = get_exchange()
    try:
        if symbol in ex.markets:
            return ex.markets[symbol]

        # symbol may be pybit-style 'BTCUSDT' -> look up by exchange market id.
        pid = _pybit_symbol(symbol)
        candidates = ex.markets_by_id.get(pid) or ex.markets_by_id.get(symbol)
        if candidates:
            if isinstance(candidates, list):
                for m in candidates:
                    if m.get("swap") and m.get("linear"):
                        return m
                return candidates[0]
            return candidates

        logger.error(f"No ccxt market found for {symbol} (id={pid})")
        return None
    except Exception as e:
        logger.error(f"Failed to load market info for {symbol}: {e}")
        return None


def calculate_proper_qty(symbol, entry_price, sl_price, ai_prob=50, regime="ranging", recent_drawdown=0.0):
    """
    Calculate quantity using AI-driven dynamic risk sizing.
    """
    market = get_symbol_info(symbol)
    if not market:
        logger.error(f"Could not get market info for {symbol}")
        return None

    try:
        risk_per_unit = abs(entry_price - sl_price)
        if risk_per_unit == 0:
            return None

        # Get base risk amount (0.5% of balance)
        base_risk = get_risk_amount()

        # Adjust risk using AI confidence
        adjusted_risk = base_risk * (get_ai_risk_percent(ai_prob, recent_drawdown, regime) / 0.5)

        raw_qty = adjusted_risk / risk_per_unit

        # Apply exchange precision
        step = market.get("precision", {}).get("amount", 0.001)
        qty = round(raw_qty / step) * step

        # Check limits
        min_qty = market.get("limits", {}).get("amount", {}).get("min", 0)
        max_qty = market.get("limits", {}).get("amount", {}).get("max", 999999)

        if qty < min_qty:
            logger.warning(f"Quantity {qty} below minimum for {symbol}")
            return None
        if qty > max_qty:
            qty = max_qty

        return round(qty, 6)

    except Exception as e:
        logger.exception(f"Quantity calculation failed for {symbol}: {e}")
        return None


def set_leverage_if_needed(symbol, desired_leverage=10):
    """Set leverage via the pybit API.

    FIX: previously used ccxt-style calls (`client.fetch_positions([...])` and
    `client.set_leverage(lev, symbol, params=...)`) on the pybit client, which
    doesn't have those. Now uses pybit's `set_leverage(category, symbol,
    buyLeverage, sellLeverage)` and treats Bybit's "leverage not modified"
    (retCode 110043) as success rather than a failure that aborts the order.
    """
    client = get_trade_client()
    sym = _pybit_symbol(symbol)
    try:
        client.set_leverage(
            category=CATEGORY,
            symbol=sym,
            buyLeverage=str(desired_leverage),
            sellLeverage=str(desired_leverage),
        )
        time.sleep(0.3)
        return True
    except Exception as e:
        # 110043 = leverage not modified (already at desired value) -> fine.
        if "110043" in str(e) or "not modified" in str(e).lower():
            return True
        logger.error(f"Leverage setting failed for {sym}: {e}")
        return False


async def execute_trade(signal):
    if not EXECUTE_TRADES:
        logger.info(f"⏸️ EXECUTE_TRADES=false — skipping order for {signal.get('symbol')} (no Bybit call made)")
        return None

    client = get_trade_client()

    try:
        symbol = signal["symbol"].split(":")[0].replace("/", "").upper()
        direction = signal["direction"]
        entry = float(signal["entry"])
        sl = float(signal.get("sl", 0))
        tp = float(signal.get("tp", 0))
        ai_prob = float(signal.get("ai_prob", 50))
        regime = signal.get("market_regime", "ranging")

        # Calculate quantity using AI risk sizing
        qty = calculate_proper_qty(symbol, entry, sl, ai_prob=ai_prob, regime=regime)
        if not qty:
            logger.error(f"❌ Invalid quantity for {symbol}")
            return None

        # Set leverage
        if not set_leverage_if_needed(symbol):
            logger.error(f"❌ Could not set leverage for {symbol}")
            return None

        side = "Buy" if direction == "LONG" else "Sell"

        # Limit entries: rest at the retracement level instead of chasing the
        # displacement candle's close at market. signal["entry"] already holds
        # the limit price when the caller runs in limit mode.
        is_limit = signal.get("entry_type") == "limit"

        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Limit" if is_limit else "Market",
            "qty": str(qty),
        }

        if is_limit:
            params["price"] = str(entry)
            params["timeInForce"] = "GTC"

        if sl:
            params["stopLoss"] = str(sl)
        if tp:
            params["takeProfit"] = str(tp)

        # Idempotency key: the SAME orderLinkId is reused across retries, so if
        # an attempt actually reached Bybit but the response timed out, the
        # retry is rejected as a duplicate instead of opening a SECOND
        # position (this was the "some trades come double" bug).
        order_link_id = f"smcw-{signal.get('trade_no', 0)}-{uuid.uuid4().hex[:10]}"
        params["orderLinkId"] = order_link_id

        logger.info(f"📤 Sending order: {params}")

        result = None
        for attempt in range(3):
            try:
                result = client.place_order(**params)
                break
            except Exception as e:
                msg = str(e).lower()
                if "orderlinkid" in msg and ("duplicate" in msg or "exist" in msg):
                    # The earlier attempt DID land — recover its orderId
                    # instead of raising or re-placing.
                    logger.info(f"↩️ Order already placed (duplicate orderLinkId), recovering {symbol}")
                    resp = client.get_open_orders(
                        category=CATEGORY, symbol=symbol, orderLinkId=order_link_id
                    )
                    rows = resp.get("result", {}).get("list", [])
                    order_id = rows[0].get("orderId") if rows else None
                    result = {"retCode": 0, "result": {"orderId": order_id,
                                                       "orderLinkId": order_link_id}}
                    break
                if attempt == 2:
                    raise e
                logger.warning(f"Retrying order ({attempt + 1}/3)")
                time.sleep(1.5)

        logger.info(f"📥 Bybit response: {result}")

        if result and result.get("retCode") == 0:
            # Write the EXECUTED qty back onto the trade record. The caller
            # sized qty with the paper formula; Bybit got THIS qty — every
            # close alert / PnL / R calculation must use the real one, or a
            # trailing-stop win shows a sliver of the actual profit.
            signal["qty"] = float(qty)
            logger.info(f"✅ Order placed: {symbol} | Qty: {qty} | AI Risk Adjusted")

        return result

    except Exception as e:
        logger.exception(f"❌ Trade execution failed: {e}")
        return None


def get_order_status(symbol, order_id):
    """Return Bybit's order status string ("New", "PartiallyFilled", "Filled",
    "Cancelled", ...) for a limit order, or None if it can't be determined.
    Checks realtime open orders first, then order history for terminal states."""
    client = get_trade_client()
    sym = _pybit_symbol(symbol)
    try:
        resp = client.get_open_orders(category=CATEGORY, symbol=sym, orderId=order_id)
        rows = resp.get("result", {}).get("list", [])
        if rows:
            return rows[0].get("orderStatus")
        resp = client.get_order_history(category=CATEGORY, symbol=sym, orderId=order_id, limit=1)
        rows = resp.get("result", {}).get("list", [])
        if rows:
            return rows[0].get("orderStatus")
        return None
    except Exception as e:
        logger.error(f"Failed to fetch order status for {sym}/{order_id}: {e}")
        return None


def cancel_order(symbol, order_id):
    """Cancel a resting limit order. Treats 'order not exists or too late'
    (already filled/cancelled) as success — the monitor re-checks status."""
    client = get_trade_client()
    sym = _pybit_symbol(symbol)
    try:
        client.cancel_order(category=CATEGORY, symbol=sym, orderId=order_id)
        logger.info(f"🚫 Cancelled limit order {order_id} on {sym}")
        return True
    except Exception as e:
        if "110001" in str(e) or "not exists" in str(e).lower():
            return True
        logger.error(f"Failed to cancel order {order_id} on {sym}: {e}")
        return False


def get_open_position_size(symbol):
    """Ask the exchange directly whether this position is still open, and if
    so, how large. Local trade records can go stale if Bybit auto-closes a
    position via its own SL/TP before our monitor loop notices.

    FIX: used ccxt `fetch_positions` on the pybit client (doesn't exist). Now
    uses pybit `get_positions(category, symbol)`.
    """
    client = get_trade_client()
    sym = _pybit_symbol(symbol)
    try:
        resp = client.get_positions(category=CATEGORY, symbol=sym)
        rows = resp.get("result", {}).get("list", [])
        for pos in rows:
            size = float(pos.get("size") or 0)
            if size > 0:
                return size
        return 0.0
    except Exception as e:
        logger.error(f"Failed to fetch live position size for {sym}: {e}")
        return None  # None = "couldn't check", distinct from 0 = "confirmed closed"


def get_last_closed_pnl(symbol):
    """
    Once a trailing stop has actually closed a position, pull the real exit
    price and realized PnL from Bybit's closed-PnL record instead of
    guessing from the last polled ticker price. Best-effort: if this fails
    (API surface differs, permissions, etc.) callers should fall back to an
    approximate price rather than blocking the trade from closing locally.
    """
    client = get_trade_client()
    try:
        symbol = symbol.split(":")[0].replace("/", "").upper()
        resp = client.get_closed_pnl(category="linear", symbol=symbol, limit=1)
        rows = resp.get("result", {}).get("list", [])
        if not rows:
            return None
        row = rows[0]
        exit_price = float(row.get("avgExitPrice", 0) or 0)
        return {
            "exit_price": exit_price if exit_price > 0 else None,
            "realized_pnl": float(row.get("closedPnl", 0) or 0),
            "qty": float(row.get("qty", 0) or 0),
        }
    except Exception as e:
        logger.debug(f"Could not fetch closed PnL for {symbol}: {e}")
        return None


def get_all_open_positions():
    """Return {pybit_symbol: size} for every currently-open linear position, or
    None if the exchange couldn't be reached. Used by reconcile.py as the
    source of truth for what is actually open."""
    client = get_trade_client()
    try:
        resp = client.get_positions(category=CATEGORY, settleCoin="USDT")
        rows = resp.get("result", {}).get("list", [])
        return {
            r["symbol"]: float(r.get("size") or 0)
            for r in rows
            if float(r.get("size") or 0) > 0
        }
    except Exception as e:
        logger.error(f"Failed to fetch open positions: {e}")
        return None


def get_wallet_balance_usdt():
    """Real account equity (USDT) from the unified wallet, or None on failure."""
    client = get_trade_client()
    try:
        resp = client.get_wallet_balance(accountType="UNIFIED")
        rows = resp.get("result", {}).get("list", [])
        if not rows:
            return None
        total = rows[0].get("totalEquity") or rows[0].get("totalWalletBalance")
        return float(total) if total not in (None, "") else None
    except Exception as e:
        logger.error(f"Failed to fetch wallet balance: {e}")
        return None


def round_price(price, market):
    """Round a price to the symbol's tick size (Bybit rejects off-tick prices)."""
    if not market:
        return round(price, 6)
    tick = market.get("precision", {}).get("price")
    if tick and tick < 1:  # ccxt gives the tick size as a float step
        return round(round(price / tick) * tick, 10)
    if tick and tick >= 1:  # some markets express precision as decimal places
        return round(price, int(tick))
    return round(price, 6)


async def activate_trailing_stop(symbol, current_price, trail_percent=0.5, trail_distance=None):
    """Let a winner run: cancel the hard take-profit and attach a trailing stop.

    Bybit's `trailingStop` is an ABSOLUTE price distance. The caller now passes
    `trail_distance` in price directly (computed ATR-aware in trade_monitor so
    small retraces don't stop the trade out). `trail_percent` remains a
    fallback: price * pct/100 when no explicit distance is given.

    Also removes the take-profit attached at entry (takeProfit="0" in the same
    set_trading_stop call) so Bybit's hard TP can't close the position at TP
    before the trailing stop ever engages.
    """
    if not EXECUTE_TRADES:
        logger.info(f"⏸️ EXECUTE_TRADES=false — skipping trailing stop for {symbol} (no Bybit call made)")
        return None

    client = get_trade_client()
    sym = _pybit_symbol(symbol)
    try:
        live_size = get_open_position_size(symbol)
        if live_size is None:
            logger.warning(f"Could not verify {sym} position size; skipping trailing activation")
            return None
        if live_size == 0:
            logger.info(f"{sym} already closed exchange-side; skipping trailing activation")
            return None

        market = get_symbol_info(symbol)
        raw_distance = trail_distance if trail_distance else current_price * trail_percent / 100.0
        distance = round_price(raw_distance, market)
        if distance <= 0:
            logger.warning(f"{sym}: computed trailing distance {distance} <= 0; skipping")
            return None

        resp = client.set_trading_stop(
            category=CATEGORY,
            symbol=sym,
            positionIdx=0,                 # one-way position mode
            takeProfit="0",                # cancel the hard TP so price can run
            trailingStop=str(distance),
            activePrice=str(round_price(current_price, market)),
        )
        pct = distance / current_price * 100 if current_price else 0
        logger.info(
            f"🚀 {sym}: TP cancelled, trailing stop set at distance={distance} "
            f"(~{pct:.2f}% of price) from {current_price}"
        )
        return resp

    except Exception as e:
        logger.exception(f"Trailing stop activation failed for {sym}: {e}")
        return None
