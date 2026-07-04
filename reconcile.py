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

from trade_manager import get_open_trades, close_trade, get_balance, save_balance
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
        # exchange-side. Pull the real exit price + realized PnL.
        info = get_last_closed_pnl(trade.get("symbol", "")) or {}
        exit_price = info.get("exit_price") or float(trade.get("tp") or trade.get("entry") or 0)
        realized = info.get("realized_pnl")
        status = "WIN" if (realized is not None and realized > 0) else "LOSS"

        logger.info(
            f"🔄 Reconcile: {sym} closed exchange-side "
            f"(exit~{exit_price:.6f}, pnl={realized}); recording locally"
        )

        close_trade(
            trade.get("symbol"),
            exit_price,
            status,
            extra_fields={
                "pnl": round(float(realized), 2) if realized is not None else None,
                "exit_reason": "Reconciled (closed on exchange)",
            },
        )

        balance = get_balance().get("balance", 0)
        try:
            await send_alert(
                format_close_alert(
                    {**trade, "exit_fee": 0},
                    exit_price,
                    "Reconciled (closed on exchange)",
                    float(realized) if realized is not None else 0.0,
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
    """Run a full reconciliation pass (positions then balance)."""
    try:
        await reconcile_positions()
        reconcile_balance()
    except Exception as e:
        logger.exception(f"Reconcile failed: {e}")
