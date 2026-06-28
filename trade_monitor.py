import logging
from exchange import get_exchange
from trade_manager import (
    get_open_trades,
    save_open_trades,
    close_trade,
    update_balance,
    get_balance
)
from telegram_alerts import send_alert

logger = logging.getLogger(__name__)
exchange = get_exchange()


async def get_current_price(symbol):
    try:

        ticker = exchange.fetch_ticker(symbol)

        return float(
            ticker.get("last", 0)
        )

    except Exception as e:

        logger.error(
            f"Price error {symbol}: {e}"
        )

        return None


async def monitor_trades():
    trades = get_open_trades()

    if not trades:

        return

    modified = False

    logger.info(
        f"Monitoring {len(trades)} trades"
    )

    for trade in trades:

        if trade.get("status") != "OPEN":

            continue

        symbol = trade["symbol"]

        entry = float(trade["entry"])

        sl = float(trade["sl"])

        tp = float(trade["tp"])

        qty = float(
            trade.get("qty", 1)
        )

        direction = trade["direction"]

        trade_no = trade.get(
            "trade_no",
            0
        )

        current_price = await get_current_price(
            symbol
        )

        if not current_price:

            continue

        #
        # BREAK EVEN
        #

        if not trade.get(
            "be_active",
            False
        ):

            if direction == "LONG":

                halfway = (
                    entry
                    + ((tp - entry) * 0.5)
                )

                if current_price >= halfway:

                    trade["sl"] = entry

                    trade["be_active"] = True

                    modified = True

                    await send_alert(
                        f"🟡 #{trade_no}\n\n"
                        f"{symbol}\n\n"
                        f"BE ACTIVE\n"
                        f"SL → ENTRY"
                    )

            else:

                halfway = (
                    entry
                    - ((entry - tp) * 0.5)
                )

                if current_price <= halfway:

                    trade["sl"] = entry

                    trade["be_active"] = True

                    modified = True

                    await send_alert(
                        f"🟡 #{trade_no}\n\n"
                        f"{symbol}\n\n"
                        f"BE ACTIVE\n"
                        f"SL → ENTRY"
                    )

        #
        # TRAILING
        #

        if not trade.get(
            "trail_active",
            False
        ):

            if direction == "LONG":

                trigger = (
                    entry
                    + ((tp - entry) * 0.75)
                )

                if current_price >= trigger:

                    trade[
                        "trail_active"
                    ] = True

                    modified = True

                    await send_alert(
                        f"🚀 #{trade_no}\n\n"
                        f"{symbol}\n\n"
                        f"TRAILING ACTIVE"
                    )

            else:

                trigger = (
                    entry
                    - ((entry - tp) * 0.75)
                )

                if current_price <= trigger:

                    trade[
                        "trail_active"
                    ] = True

                    modified = True

                    await send_alert(
                        f"🚀 #{trade_no}\n\n"
                        f"{symbol}\n\n"
                        f"TRAILING ACTIVE"
                    )

        #
        # MOVE TRAILING SL
        #

        if trade.get(
            "trail_active",
            False
        ):

            if direction == "LONG":

                new_sl = (
                    current_price * 0.995
                )

                if new_sl > trade["sl"]:

                    trade["sl"] = new_sl

                    modified = True

            else:

                new_sl = (
                    current_price * 1.005
                )

                if new_sl < trade["sl"]:

                    trade["sl"] = new_sl

                    modified = True

        #
        # CHECK TP / SL
        #

        sl = float(trade["sl"])

        if direction == "LONG":

            pnl = (
                current_price - entry
            ) * qty

            hit_tp = (
                current_price >= tp
            )

            hit_sl = (
                current_price <= sl
            )

        else:

            pnl = (
                entry - current_price
            ) * qty

            hit_tp = (
                current_price <= tp
            )

            hit_sl = (
                current_price >= sl
            )

        if hit_tp or hit_sl:

            result = (
                "WIN"
                if hit_tp
                else "LOSS"
            )

            trade["status"] = "CLOSED"

            close_trade(
                symbol,
                current_price,
                result
            )

            update_balance(
                pnl
            )

            balance = get_balance()[
                "balance"
            ]

            await send_alert(
                f"{'✅' if hit_tp else '❌'} "
                f"#{trade_no}\n\n"
                f"{symbol}\n\n"
                f"{result}\n\n"
                f"Entry {entry:.6f}\n"
                f"Exit {current_price:.6f}\n\n"
                f"{'+' if pnl >= 0 else ''}"
                f"${pnl:.2f}\n\n"
                f"💰 ${balance:.2f}"
            )

            modified = True

    if modified:

        save_open_trades(
            trades
        )
