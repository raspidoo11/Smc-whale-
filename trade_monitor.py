import logging
import asyncio
from datetime import datetime, timezone
from trade_manager import get_open_trades, save_open_trades, get_balance
from exchange import get_exchange
from bybit_executor import (
    EXECUTE_TRADES,
    activate_trailing_stop,
    close_position_market,
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
from config import (
    TRAIL_PERCENT,
    TRAIL_ACTIVATION_RATIO,
    LIMIT_TTL_MINUTES,
    INVALIDATE_PENDING_ON_STRUCTURE,
    MAX_HOLD_MINUTES,
)

logger = logging.getLogger(__name__)
exchange = get_exchange()

# Live-only: how long to wait before retrying a failed trailing-stop
# activation, and how many attempts before force-closing so a trade can't get
# stuck open (and missing from training data) forever.
RETRY_COOLDOWN_MINUTES = 1
MAX_ACTIVATION_ATTEMPTS = 2


def _hold_expired(trade):
    """Scalp pacing: has this OPEN trade overstayed MAX_HOLD_MINUTES?
    Trades already running on a trailing stop are exempt — a winner being
    trailed is never evicted; the time stop only clears DEAD trades that
    neither hit SL nor reached the trail zone."""
    if MAX_HOLD_MINUTES <= 0 or trade.get("trailing_stop_active"):
        return False
    held = _minutes_since(trade.get("filled_at") or trade.get("placed_at"))
    return held is not None and held > MAX_HOLD_MINUTES


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

    Returns net pnl (float) on success, or None if the trade was already
    closed elsewhere (caller must NOT drop it from the in-memory list in
    that case without checking — usually it is already gone on disk).
    """
    from paper_trader import close_paper_trade_with_fees
    pnl_after_fees = close_paper_trade_with_fees(trade, exit_price, exit_reason)

    # None = the trade was already closed by another path (e.g. reconcile) —
    # balance untouched, and no second close alert.
    if pnl_after_fees is None:
        return None

    balance = get_balance()["balance"]
    await send_alert(format_close_alert(trade, exit_price, exit_reason, pnl_after_fees, balance))
    # Retraining is intentionally NOT triggered here — it runs on its own
    # 10-minute schedule in main.py, decoupled from the monitor hot path.
    return pnl_after_fees


def _fee_aware_breakeven(direction, entry, fee_rate):
    """Minimum favorable exit so net PnL after a single-side fee is still >= 0.

    Paper close math is (exit - entry)*qty - exit*fee for LONG (and the
    symmetric form for SHORT). Solving for exit gives a tiny buffer above
    (LONG) / below (SHORT) entry — enough that a trail fill never flips to
    LOSS purely from fee rounding after the trade already reached the arm zone.
    """
    entry = float(entry)
    fee_rate = float(fee_rate)
    if fee_rate <= 0 or fee_rate >= 1:
        return entry
    if direction == "LONG":
        return entry / (1.0 - fee_rate)
    return entry / (1.0 + fee_rate)


def trail_stop_price(direction, anchor, trail_percent, entry, fee_rate=0.0):
    """Ratcheting trail stop with profit lock + fee-aware breakeven floor.

    Once trailing is armed the trade has already travelled ~TRAIL_ACTIVATION_RATIO
    of the way to TP. A pure percent-of-price trail is often *wider* than the
    locked-in scalp progress (e.g. 0.3% trail vs 0.27% open profit at arm),
    which pinned every paper trail exit to pure breakeven and credited ~$0
    to the paper balance even on "WIN" trail hits.

    Fix: trail distance = min(pct-of-price, 50% of open profit), then floor
    at fee-aware BE so an armed trail never becomes a LOSS.
    """
    trail = float(trail_percent)
    anchor = float(anchor)
    entry = float(entry)
    be = _fee_aware_breakeven(direction, entry, fee_rate)

    pct_dist = abs(anchor) * trail / 100.0
    open_profit = abs(anchor - entry)
    if open_profit > 0:
        dist = min(pct_dist, open_profit * 0.5)
    else:
        dist = pct_dist

    if direction == "LONG":
        raw = anchor - dist
        return max(raw, be)
    raw = anchor + dist
    return min(raw, be)


async def _handle_paper_trailing(trade, symbol, direction, current_price, open_trades):
    """Simulate a ratcheting trailing stop for paper trades.

    Previously paper mode had no real trailing stop at all: activate_trailing_
    stop() short-circuits to None when EXECUTE_TRADES=false, so the monitor
    logged two 'activation failed' errors and force-closed the trade at market
    with the ugly reason 'Trailing Stop Failed - Forced Close'. Now the trail
    anchor ratchets with favorable price and the trade closes when price
    retraces from the best level. Returns True if the trade was closed and
    removed from the in-memory list."""
    from paper_trader import FEE_RATE

    trail = float(trade.get("trail_percent", TRAIL_PERCENT))
    anchor = float(trade.get("trail_anchor", current_price))
    entry = float(trade.get("entry", 0) or 0)

    if direction == "LONG":
        anchor = max(anchor, current_price)
        stop_price = trail_stop_price("LONG", anchor, trail, entry, FEE_RATE)
        breached = current_price <= stop_price
    else:
        anchor = min(anchor, current_price)
        stop_price = trail_stop_price("SHORT", anchor, trail, entry, FEE_RATE)
        breached = current_price >= stop_price

    if breached:
        logger.info(f"✅ {symbol} paper trailing stop hit at ~{stop_price:.6f}")
        # Close FIRST (credits paper balance atomically). Only then drop from
        # the in-memory list — removing first and then failing close left
        # trades gone from open_trades with no history and no balance credit.
        pnl = await _close_trade_record(trade, stop_price, "Trailing Stop Hit")
        if pnl is not None and trade in open_trades:
            open_trades.remove(trade)
            return True
        if pnl is not None:
            return True
        return False

    # Ratchet the anchor forward; caller persists via trades_changed.
    trade["trail_anchor"] = anchor
    trade["trail_stop"] = stop_price
    return False


def _structure_invalidated(direction, current_price, trade):
    """True if price has traded through the setup's invalidation level.

    Prefer explicit invalidation_price / structure_swing (the raw swing the
    stop was anchored to). Fall back to SL only when those are missing — and
    only when SL is on the correct side of the resting limit, so a bad fixture
    never false-cancels.
    """
    if not INVALIDATE_PENDING_ON_STRUCTURE:
        return False
    if current_price is None:
        return False

    level = trade.get("invalidation_price")
    if level is None:
        level = trade.get("structure_swing")
    if level is None:
        level = trade.get("sl")
        # Guard: SL must be on the protective side of the limit entry.
        entry = trade.get("entry")
        if level is None or entry is None:
            return False
        try:
            if direction == "LONG" and float(level) >= float(entry):
                return False
            if direction == "SHORT" and float(level) <= float(entry):
                return False
        except (TypeError, ValueError):
            return False

    try:
        level = float(level)
        price = float(current_price)
    except (TypeError, ValueError):
        return False

    if direction == "LONG":
        return price < level
    return price > level


async def _handle_pending_order(trade, symbol, direction, current_price, open_trades):
    """Lifecycle of a resting limit order (status == "PENDING").

    Paper: fill when price trades through the limit level; expire after
    LIMIT_TTL_MINUTES if never touched; cancel early if structure invalidates.
    Live: mirror Bybit's actual order status, cancelling on expiry or
    structure break. Cancelled/expired orders are removed WITHOUT touching
    trade history — a never-opened order must not pollute training data.
    Returns True if the pending entry left the book (filled or cancelled)."""
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
    invalidated = _structure_invalidated(direction, current_price, trade)

    if not EXECUTE_TRADES:
        # Structure break first: do not fill a thesis that already failed.
        if invalidated:
            await _cancel("Structure invalidated before fill")
            return True
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

    if invalidated:
        if cancel_order(symbol, order_id):
            if get_order_status(symbol, order_id) == "Filled":
                await _fill()
            else:
                await _cancel("Structure invalidated before fill")
            return True

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

        if not open_trades:
            return  # nothing to do — and no log line every 35s saying so

        logger.info(f"📊 Monitoring {len(open_trades)} open trade(s)")

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

                # Per-trade price ticks are DEBUG: with 10 open trades this
                # printed ~17 lines/min at INFO and added nothing actionable —
                # state CHANGES (fills, closes, trailing arms) log at INFO.
                logger.debug(
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
                        pnl = await _close_trade_record(trade, exit_price, exit_reason)
                        if pnl is not None and trade in open_trades:
                            open_trades.remove(trade)
                            trades_changed = True
                        elif pnl is not None:
                            trades_changed = True
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

                # ---- Time stop (scalp pacing): evict trades that have gone
                # nowhere for MAX_HOLD_MINUTES. SL and trail-arming take
                # priority; live closes must succeed on the exchange first.
                if not hit_sl and not near_tp and _hold_expired(trade):
                    if not await close_position_market(symbol, direction, qty):
                        continue  # couldn't verify exchange close — retry next cycle
                    logger.info(f"⏱️ Time stop on {symbol} after {MAX_HOLD_MINUTES:.0f} min")
                    pnl = await _close_trade_record(trade, current_price, "Time Stop (max hold)")
                    if pnl is not None and trade in open_trades:
                        open_trades.remove(trade)
                        trades_changed = True
                    elif pnl is not None:
                        trades_changed = True
                    continue

                if hit_sl:
                    exit_price = sl
                    logger.info(f"🚨 Stop Loss Hit on {symbol} at ${exit_price:.6f}")
                    pnl = await _close_trade_record(trade, exit_price, "Stop Loss Hit")
                    if pnl is not None and trade in open_trades:
                        open_trades.remove(trade)
                        trades_changed = True
                    elif pnl is not None:
                        trades_changed = True
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
                        pnl = await _close_trade_record(
                            trade, current_price, "Trailing Stop Failed - Forced Close"
                        )
                        if pnl is not None and trade in open_trades:
                            open_trades.remove(trade)
                            trades_changed = True
                        elif pnl is not None:
                            trades_changed = True
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
