"""Exchange reconciliation (live/demo only).

In live trading the exchange — not our local JSON — is the source of truth.
Bybit can close a position via the SL/TP attached at entry before our 35s
monitor loop ever notices, leaving a phantom "OPEN" trade locally that never
gets recorded to history (and so never trains the model). Local balance can
likewise drift from real equity because the monitor updates it with a
*simulated* PnL.

This module periodically:
  * closes local trades that no longer exist on the exchange, recording the
    real exit price / realized PnL from Bybit's closed-PnL endpoint, and
  * snaps local balance to real wallet equity.

It is a no-op in paper mode, where the local monitor is authoritative.
"""

import logging

from trade_manager import (
    get_open_trades,
    close_trade,
    get_balance,
    save_balance,
    add_daily_pnl,
)
from bybit_executor import (
    EXECUTE_TRADES,
    _pybit_symbol,
    get_all_open_positions,
    get_last_closed_pnl,
    get_wallet_balance_usdt,
)
from telegram_alerts import send_alert
from alerts import format_close_alert

logger = logging.getLogger(__name__)


def _fallback_pnl(trade, exit_price):
    """Compute PnL from our own trade record when Bybit's closed-PnL endpoint
    returns nothing in time. Previously that case fell back to 0.0, so the
    close alert showed 'LOSS +0.00 USDT' no matter the real outcome — the
    W/L and profit weren't missing from the alert, they were being fed a
    zero. Approximate (no fee detail), but correct in sign and magnitude."""
    entry = float(trade.get("entry") or 0)
    qty = float(trade.get("qty") or 0)
    direction = str(trade.get("direction", "LONG")).upper()
    move = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
    return round(move * qty, 4)


async def reconcile_positions():
    if not EXECUTE_TRADES:
        return

    open_trades = [t for t in get_open_trades() if t.get("status") == "OPEN"]
    if not open_trades:
        return

    live = get_all_open_positions()
    if live is None:
        logger.debug("Reconcile: could not fetch live positions, skipping this pass")
        return

    for trade in open_trades:
        sym = _pybit_symbol(trade.get("symbol", ""))
        if live.get(sym, 0) > 0:
            continue  # still open on the exchange

        # Position is gone on Bybit but still OPEN locally -> it closed
        # exchange-side. Pull the real exit price + realized PnL; if the
        # closed-PnL endpoint has nothing yet, compute PnL from our own
        # record so the alert and history NEVER carry a blank/zero result.
        info = get_last_closed_pnl(trade.get("symbol", "")) or {}
        exit_price = info.get("exit_price") or float(trade.get("tp") or trade.get("entry") or 0)
        realized = info.get("realized_pnl")
        approx = realized is None
        if approx:
            realized = _fallback_pnl(trade, exit_price)
        realized = float(realized)
        status = "WIN" if realized > 0 else "LOSS"

        logger.info(
            f"🔄 Reconcile: {sym} closed exchange-side "
            f"(exit~{exit_price:.6f}, {status} pnl={realized:+.2f}"
            f"{' approx' if approx else ''}); recording locally"
        )

        closed = close_trade(
            trade.get("symbol"),
            exit_price,
            status,
            extra_fields={
                "pnl": round(realized, 2),
                "exit_reason": "Closed on exchange (SL/TP)"
                               + (" · approx PnL" if approx else ""),
            },
        )

        if closed is None:
            continue  # another path already recorded it — no daily-pnl/alert

        # Balance is synced from wallet equity (reconcile_balance runs first),
        # but the daily circuit breaker must still see this loss/win — without
        # this, every exchange-side close was invisible to DAILY_LOSS_LIMIT.
        add_daily_pnl(realized)

        balance = get_balance().get("balance", 0)
        try:
            await send_alert(
                format_close_alert(
                    {**trade, "exit_fee": 0},
                    exit_price,
                    "Closed on exchange (SL/TP)",
                    realized,
                    balance,
                )
            )
        except Exception as e:
            logger.debug(f"Reconcile alert failed for {sym}: {e}")


def reconcile_balance():
    if not EXECUTE_TRADES:
        return
    equity = get_wallet_balance_usdt()
    if equity is None:
        return
    data = get_balance()
    if abs(float(data.get("balance", 0)) - equity) > 1e-6:
        logger.info(f"🔄 Reconcile: balance {data.get('balance')} -> real equity {equity}")
        data["balance"] = equity
        save_balance(data)


async def reconcile():
    """Run a full reconciliation pass. Balance FIRST: wallet equity already
    reflects any exchange-side closes, so syncing it before recording those
    closes means the close alert's 'Balance' line shows the fresh number
    instead of the stale pre-close one."""
    try:
        reconcile_balance()
        await reconcile_positions()
    except Exception as e:
        logger.exception(f"Reconcile failed: {e}")
