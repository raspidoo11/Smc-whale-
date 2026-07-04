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
)
from telegram_alerts import send_alert
from alerts import format_close_alert, format_trailing_alert

logger = logging.getLogger(__name__)
exchange = get_exchange()

# Trailing-stop trail distance (percent). Used for both the live Bybit trailing
# stop and the paper-mode simulation.
TRAIL_PERCENT = 0.5

# Live-only: how long to wait before retrying a failed trailing-stop
# activation, and how many attempts before force-closing so a trade can't get
# stuck open (and missing from training data) forever.
RETRY_COOLDOWN_MINUTES = 1
MAX_ACTIVATION_ATTEMPTS = 2


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
                    hit_tp = current_price >= tp
                    hit_sl = current_price <= sl
                elif direction == "SHORT":
                    hit_tp = current_price <= tp
                    hit_sl = current_price >= sl
                else:
                    continue

                if hit_sl:
                    exit_price = sl
                    logger.info(f"🚨 Stop Loss Hit on {symbol} at ${exit_price:.6f}")
                    open_trades.remove(trade)
                    trades_changed = True
                    await _close_trade_record(trade, exit_price, "Stop Loss Hit")
                    continue

                if hit_tp:
                    # ---- Paper: arm the simulated trailing stop ----
                    if not EXECUTE_TRADES:
                        trade["trailing_stop_active"] = True
                        trade["trail_anchor"] = current_price
                        trade["trail_percent"] = TRAIL_PERCENT
                        trade["trailing_stop_activated_at"] = datetime.now(timezone.utc).isoformat()
                        trades_changed = True
                        logger.info(f"🚀 {symbol} TP reached — arming paper trailing stop")
                        await send_alert(format_trailing_alert(trade, current_price, TRAIL_PERCENT))
                        continue

                    # ---- Live: activate the Bybit trailing stop (with retry) ----
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

                    logger.info(f"🚀 Activating trailing stop on {symbol} (TP reached)")
                    result = await activate_trailing_stop(
                        symbol=symbol,
                        direction=direction,
                        qty=qty,
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
