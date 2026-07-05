import logging
import asyncio
from datetime import datetime, timezone
from trade_manager import get_open_trades, save_open_trades, get_balance
from exchange import get_exchange
from bybit_executor import (
    EXECUTE_TRADES,
    activate_trailing_stop,
    get_open_position_size,
    get_last_closed_pnl,
    get_order_status,
    cancel_order,
)
from telegram_alerts import send_alert
from alerts import (
    format_close_alert,
    format_trailing_alert,
    format_limit_filled_alert,
    format_limit_cancelled_alert,
)
from config import TRAIL_PERCENT, TRAIL_ACTIVATION_RATIO, LIMIT_TTL_MINUTES

logger = logging.getLogger(__name__)
exchange = get_exchange()

# Live-only: how long to wait before retrying a failed trailing-stop
# activation, and how many attempts before force-closing so a trade can't get
# stuck open (and missing from training data) forever.
RETRY_COOLDOWN_MINUTES = 1
MAX_ACTIVATION_ATTEMPTS = 2


def tp_progress(direction, entry, tp, price):
    """Fraction of the way price has travelled from entry toward TP (0..1+).
    1.0 == TP reached. Used to arm the trailing stop slightly BEFORE TP so we
    can cancel the hard TP and let the winner run instead of being capped."""
    if direction == "LONG":
        span = tp - entry
        return (price - entry) / span if span > 0 else 0.0
    span = entry - tp
    return (entry - price) / span if span > 0 else 0.0


async def get_current_price(symbol):
    try:
        if not exchange:
            return None
        ticker = exchange.fetch_ticker(symbol)
        return ticker.get("last")
    except Exception as e:
        logger.debug(f"Failed to get price for {symbol}: {e}")
        return None


def _minutes_since(iso_timestamp):
    if not iso_timestamp:
        return None
    try:
        last = datetime.fromisoformat(iso_timestamp)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - last).total_seconds() / 60
    except Exception:
        return None


async def _close_trade_record(trade, exit_price, exit_reason):
    """Shared close path so trailing-stop closes get recorded exactly like SL
    closes: same balance update, same trade_history write, same alert style.
    Relies on close_paper_trade_with_fees internally calling
    trade_manager.close_trade()."""
    from paper_trader import close_paper_trade_with_fees
    pnl_after_fees = close_paper_trade_with_fees(trade, exit_price, exit_reason)

    balance = get_balance()["balance"]
    await send_alert(format_close_alert(trade, exit_price, exit_reason, pnl_after_fees, balance))
    # Retraining is intentionally NOT triggered here — it runs on its own
    # 10-minute schedule in main.py, decoupled from the monitor hot path.


async def _handle_paper_trailing(trade, symbol, direction, current_price, open_trades):
    """Simulate a ratcheting trailing stop for paper trades.

    Previously paper mode had no real trailing stop at all: activate_trailing_
    stop() short-circuits to None when EXECUTE_TRADES=false, so the monitor
    logged two 'activation failed' errors and force-closed the trade at market
    with the ugly reason 'Trailing Stop Failed - Forced Close'. Now the trail
    anchor ratchets with favorable price and the trade closes when price
    retraces TRAIL_PERCENT from the best level — the same behavior the live
    Bybit trailing stop provides. Returns True if the trade was closed."""
    trail = float(trade.get("trail_percent", TRAIL_PERCENT))
    anchor = float(trade.get("trail_anchor", current_price))

    if direction == "LONG":
        anchor = max(anchor, current_price)
        stop_price = anchor * (1 - trail / 100)
        breached = current_price <= stop_price
    else:
        anchor = min(anchor, current_price)
        stop_price = anchor * (1 + trail / 100)
        breached = current_price >= stop_price

    if breached:
        logger.info(f"✅ {symbol} paper trailing stop hit at ~{stop_price:.6f}")
        open_trades.remove(trade)
        await _close_trade_record(trade, stop_price, "Trailing Stop Hit")
        return True

    # Ratchet the anchor forward; caller persists via trades_changed.
    trade["trail_anchor"] = anchor
    return False


async def _handle_pending_order(trade, symbol, direction, current_price, open_trades):
    """Lifecycle of a resting limit order (status == "PENDING").

    Paper: fill when price trades through the limit level; expire after
    LIMIT_TTL_MINUTES if never touched. Live: mirror Bybit's actual order
    status, cancelling the resting order on expiry. Cancelled/expired orders
    are removed WITHOUT touching trade history — a never-opened order must not
    pollute training data. Returns True if the pending entry left the book
    (filled or cancelled) so the caller persists."""
    limit_price = float(trade.get("entry", 0))

    async def _fill():
        trade["status"] = "OPEN"
        trade["filled_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(f"✅ Limit FILLED: {trade.get('symbol')} at {limit_price:.6f}")
        await send_alert(format_limit_filled_alert(trade))

    async def _cancel(reason):
        open_trades.remove(trade)
        logger.info(f"🚫 Limit cancelled ({reason}): {trade.get('symbol')}")
        await send_alert(format_limit_cancelled_alert(trade, reason))

    minutes_pending = _minutes_since(trade.get("placed_at"))
    expired = minutes_pending is not None and minutes_pending > LIMIT_TTL_MINUTES

    if not EXECUTE_TRADES:
        touched = (
            current_price <= limit_price
            if direction == "LONG"
            else current_price >= limit_price
        )
        if touched:
            await _fill()
        elif expired:
            await _cancel(f"Not filled within {int(LIMIT_TTL_MINUTES)} min")
        return touched or expired

    # ---- Live: Bybit's order state is the truth ----
    order_id = trade.get("order_id")
    if not order_id:
        await _cancel("No exchange order id recorded")
        return True

    status = get_order_status(symbol, order_id)

    if status == "Filled":
        await _fill()
        return True

    if status in ("Cancelled", "Rejected", "Deactivated"):
        await _cancel(f"Order {status} on exchange")
        return True

    if status == "PartiallyFilled":
        # A position already exists; let it keep filling — never expire a
        # partially-filled order out from under an open position.
        return False

    if expired:
        if cancel_order(symbol, order_id):
            # Re-check: it may have filled in the race window before cancel.
            if get_order_status(symbol, order_id) == "Filled":
                await _fill()
            else:
                await _cancel(f"Not filled within {int(LIMIT_TTL_MINUTES)} min")
            return True
    return False


async def monitor_trades():
    try:
        open_trades = get_open_trades()
        logger.info(f"📊 Monitoring {len(open_trades)} open trade(s)")

        if not open_trades:
            return

        trades_changed = False

        # Iterate over a copy -- we may remove closed trades as we go.
        for trade in open_trades[:]:
            try:
                symbol = trade.get("symbol")
                if not symbol:
                    continue

                current_price = await get_current_price(symbol)
                if current_price is None:
                    continue

                entry = float(trade.get("entry", 0))
                sl = float(trade.get("sl", 0))
                tp = float(trade.get("tp", 0))
                direction = trade.get("direction")
                qty = float(trade.get("qty", 0))

                if not all([entry, sl, tp, direction, qty]):
                    continue

                logger.info(
                    f"{symbol} | Current={current_price:.6f} | TP={tp:.6f} | SL={sl:.6f}"
                )

                # ============================================================
                # Resting limit order: wait for fill or expiry
                # ============================================================
                if trade.get("status") == "PENDING":
                    changed = await _handle_pending_order(
                        trade, symbol, direction, current_price, open_trades
                    )
                    if changed:
                        trades_changed = True
                    continue

                # ============================================================
                # Trailing stop already active
                # ============================================================
                if trade.get("trailing_stop_active"):
                    if not EXECUTE_TRADES:
                        closed = await _handle_paper_trailing(
                            trade, symbol, direction, current_price, open_trades
                        )
                        trades_changed = True
                        continue

                    # ---- Live: poll whether Bybit has closed the position ----
                    live_size = get_open_position_size(symbol)
                    if live_size is None:
                        continue  # couldn't verify, retry next cycle
                    if live_size == 0:
                        closed_info = get_last_closed_pnl(symbol)
                        if closed_info and closed_info.get("exit_price"):
                            exit_price = closed_info["exit_price"]
                            exit_reason = "Trailing Stop Hit"
                        else:
                            exit_price = current_price
                            exit_reason = "Trailing Stop Hit (approx exit price)"
                        logger.info(f"✅ {symbol} closed via trailing stop at ~${exit_price:.6f}")
                        open_trades.remove(trade)
                        trades_changed = True
                        await _close_trade_record(trade, exit_price, exit_reason)
                    continue

                # ============================================================
                # TP / SL evaluation
                # ============================================================
                if direction == "LONG":
                    hit_sl = current_price <= sl
                elif direction == "SHORT":
                    hit_sl = current_price >= sl
                else:
                    continue

                # Arm the trailing stop when price is TRAIL_ACTIVATION_RATIO of
                # the way to TP (e.g. 97%) rather than AT tp — so we can cancel
                # the hard TP and let the winner run past it.
                near_tp = tp_progress(direction, entry, tp, current_price) >= TRAIL_ACTIVATION_RATIO

                if hit_sl:
                    exit_price = sl
                    logger.info(f"🚨 Stop Loss Hit on {symbol} at ${exit_price:.6f}")
                    open_trades.remove(trade)
                    trades_changed = True
                    await _close_trade_record(trade, exit_price, "Stop Loss Hit")
                    continue

                if near_tp:
                    # ---- Paper: arm the simulated trailing stop ----
                    if not EXECUTE_TRADES:
                        trade["trailing_stop_active"] = True
                        trade["trail_anchor"] = current_price
                        trade["trail_percent"] = TRAIL_PERCENT
                        trade["trailing_stop_activated_at"] = datetime.now(timezone.utc).isoformat()
                        trades_changed = True
                        logger.info(f"🚀 {symbol} near TP ({TRAIL_ACTIVATION_RATIO:.0%}) — arming paper trailing stop")
                        await send_alert(format_trailing_alert(trade, current_price, TRAIL_PERCENT))
                        continue

                    # ---- Live: cancel TP + activate the Bybit trailing stop ----
                    attempts = trade.get("trailing_stop_attempts", 0)
                    minutes_since_last = _minutes_since(trade.get("trailing_stop_last_attempt"))

                    if minutes_since_last is not None and minutes_since_last < RETRY_COOLDOWN_MINUTES:
                        continue

                    if attempts >= MAX_ACTIVATION_ATTEMPTS:
                        logger.warning(
                            f"{symbol}: trailing stop failed {attempts}x, forcing close at market"
                        )
                        open_trades.remove(trade)
                        trades_changed = True
                        await _close_trade_record(
                            trade, current_price, "Trailing Stop Failed - Forced Close"
                        )
                        continue

                    logger.info(f"🚀 Activating trailing stop on {symbol} (near TP)")
                    result = await activate_trailing_stop(
                        symbol=symbol,
                        current_price=current_price,
                        trail_percent=TRAIL_PERCENT,
                    )

                    trade["trailing_stop_attempts"] = attempts + 1
                    trade["trailing_stop_last_attempt"] = datetime.now(timezone.utc).isoformat()
                    trades_changed = True

                    if result:
                        trade["trailing_stop_active"] = True
                        trade["trailing_stop_activated_at"] = datetime.now(timezone.utc).isoformat()
                        await send_alert(format_trailing_alert(trade, current_price, TRAIL_PERCENT))
                    else:
                        logger.error(
                            f"Failed to activate trailing stop for {symbol} "
                            f"(attempt {attempts + 1}/{MAX_ACTIVATION_ATTEMPTS})"
                        )
                    continue

            except Exception as e:
                logger.exception(f"❌ Error monitoring trade {trade.get('symbol')}: {e}")

        if trades_changed:
            save_open_trades(open_trades)

    except Exception as e:
        logger.exception(f"❌ Monitor trades failed: {e}")


async def main():
    await monitor_trades()


if __name__ == "__main__":
    asyncio.run(main())
