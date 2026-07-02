import logging
import asyncio
from datetime import datetime, timezone
from trade_manager import get_open_trades, save_open_trades, get_balance
from exchange import get_exchange
from bybit_executor import activate_trailing_stop, get_open_position_size, get_last_closed_pnl
from telegram_alerts import send_alert
from xgboost_trainer import train_model_incremental

logger = logging.getLogger(__name__)
exchange = get_exchange()

# How long to wait before retrying a failed trailing-stop activation, and how
# many attempts to allow before giving up and force-closing the trade so it
# can't get stuck open (and missing from training data) forever.
RETRY_COOLDOWN_MINUTES = 5
MAX_ACTIVATION_ATTEMPTS = 5


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
    """Shared close path so trailing-stop closes get recorded the same way
    SL closes always have: same balance update, same trade_history write,
    same training trigger. Relies on close_paper_trade_with_fees internally
    calling trade_manager.close_trade(), same as the original SL branch did."""
    from paper_trader import close_paper_trade_with_fees
    pnl_after_fees = close_paper_trade_with_fees(trade, exit_price, exit_reason)

    balance = get_balance()["balance"]
    await send_alert(
        f"✅ {exit_reason}\n"
        f"{trade.get('direction')} {trade.get('symbol')}\n"
        f"Entry: ${float(trade.get('entry', 0)):.6f} → Exit: ${exit_price:.6f}\n"
        f"Qty: {trade.get('qty')}\n"
        f"Net PnL: ${pnl_after_fees:.2f}\n"
        f"Balance: ${balance:.2f}"
    )
    train_model_incremental()


async def monitor_trades():
    try:
        open_trades = get_open_trades()
        logger.info(f"📊 Monitoring {len(open_trades)} open trade(s)")

        if not open_trades:
            return

        trades_changed = False

        # Iterate over a copy -- we may remove closed trades from
        # `open_trades` as we go, and mutating a list while iterating it
        # directly silently skips entries.
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

                # ---- Trailing stop already active: watch for real closure ----
                # This is the fix for the infinite-retry bug: once activated,
                # we stop calling activate_trailing_stop() entirely and just
                # poll whether Bybit has actually closed the position yet.
                if trade.get("trailing_stop_active"):
                    live_size = get_open_position_size(symbol)

                    if live_size is None:
                        continue  # couldn't verify, try again next cycle

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

                    # Still running with the trailing stop live -- nothing to do.
                    continue

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
                    exit_reason = "Stop Loss Hit"
                    logger.info(f"🚨 {exit_reason} on {symbol} at ${exit_price:.6f}")
                    open_trades.remove(trade)
                    trades_changed = True
                    await _close_trade_record(trade, exit_price, exit_reason)
                    continue

                if hit_tp:
                    attempts = trade.get("trailing_stop_attempts", 0)
                    minutes_since_last = _minutes_since(trade.get("trailing_stop_last_attempt"))

                    if minutes_since_last is not None and minutes_since_last < RETRY_COOLDOWN_MINUTES:
                        continue  # tried recently, don't hammer the exchange every 35s

                    if attempts >= MAX_ACTIVATION_ATTEMPTS:
                        # Trailing stop keeps failing -- force a close so the
                        # trade can't sit stuck forever and vanish from
                        # training data.
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
                        trail_percent=0.5
                    )

                    trade["trailing_stop_attempts"] = attempts + 1
                    trade["trailing_stop_last_attempt"] = datetime.now(timezone.utc).isoformat()
                    trades_changed = True

                    if result:
                        trade["trailing_stop_active"] = True
                        trade["trailing_stop_activated_at"] = datetime.now(timezone.utc).isoformat()
                        await send_alert(
                            f"🚀 Trailing Stop Activated\n"
                            f"{direction} {symbol}\n"
                            f"Original TP: ${tp:.6f}\n"
                            f"Current Price: ${current_price:.6f}\n"
                            f"Trail: 0.5%"
                        )
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
