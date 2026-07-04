"""Telegram message formatting. Kept separate from the send transport
(telegram_alerts.py) so open/close/trailing alerts share one consistent,
readable house style instead of being hand-built at each call site."""

from xgboost_trainer import calculate_realized_r

BAR = "━━━━━━━━━━━━━━━━━━━━"


def _dir_badge(direction):
    return "🟢 LONG" if str(direction).upper() == "LONG" else "🔴 SHORT"


def _roi_pct(trade, exit_price):
    """Return on margin (%), i.e. leveraged ROI, sign-aware by direction."""
    entry = float(trade.get("entry", 0) or 0)
    qty = float(trade.get("qty", 0) or 0)
    lev = float(trade.get("leverage", 1) or 1)
    if entry <= 0 or qty <= 0:
        return 0.0
    direction = str(trade.get("direction", "LONG")).upper()
    move = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
    notional = entry * qty
    margin = notional / max(lev, 1)
    if margin <= 0:
        return 0.0
    return (move * qty) / margin * 100


def format_open_alert(trade):
    n = trade.get("trade_no", "?")
    conf = trade.get("confidence", "?")
    ai = trade.get("ai_prob")
    ai_line = f"\n🤖 <b>AI Prob:</b> <code>{float(ai):.1f}%</code>" if ai is not None else ""
    er = trade.get("expected_r")
    er_line = f"\n📐 <b>Exp. R:</b> <code>{float(er):+.2f}</code>" if er is not None else ""
    regime = trade.get("market_regime")
    regime_line = f"\n🌡️ <b>Regime:</b> {regime}" if regime else ""
    return (
        f"🚨 <b>SIGNAL #{n} — OPENED</b>\n"
        f"{BAR}\n"
        f"🪙 <b>{trade.get('symbol')}</b>   {_dir_badge(trade.get('direction'))}\n\n"
        f"💵 <b>Entry:</b> <code>{float(trade.get('entry', 0)):.6f}</code>\n"
        f"🛑 <b>Stop:</b> <code>{float(trade.get('sl', 0)):.6f}</code>\n"
        f"🎯 <b>Target:</b> <code>{float(trade.get('tp', 0)):.6f}</code>\n"
        f"📦 <b>Qty:</b> <code>{float(trade.get('qty', 0)):.4f}</code>\n"
        f"🔥 <b>Confidence:</b> <code>{conf}%</code>"
        f"{ai_line}{er_line}{regime_line}\n"
        f"{BAR}"
    )


def format_close_alert(trade, exit_price, exit_reason, pnl, balance):
    n = trade.get("trade_no", "?")
    is_win = pnl > 0
    head_emoji = "✅" if is_win else "🔻"
    result = "WIN" if is_win else "LOSS"
    pnl_emoji = "🟩" if is_win else "🟥"

    r_multiple = calculate_realized_r({**trade, "exit_price": exit_price})
    roi = _roi_pct(trade, exit_price)
    fees = float(trade.get("entry_fee", 0) or 0) + float(trade.get("exit_fee", 0) or 0)

    return (
        f"{head_emoji} <b>TRADE #{n} CLOSED — {result}</b>\n"
        f"{BAR}\n"
        f"🪙 <b>{trade.get('symbol')}</b>   {_dir_badge(trade.get('direction'))}\n"
        f"📍 <b>Exit:</b> {exit_reason}\n\n"
        f"💵 <b>Entry:</b> <code>{float(trade.get('entry', 0)):.6f}</code>\n"
        f"🏁 <b>Exit:</b> <code>{float(exit_price):.6f}</code>\n\n"
        f"{pnl_emoji} <b>Net PnL:</b> <code>{pnl:+.2f} USDT</code>  (<code>{roi:+.2f}%</code>)\n"
        f"📐 <b>R Multiple:</b> <code>{r_multiple:+.2f}R</code>\n"
        f"💸 <b>Fees:</b> <code>{fees:.2f} USDT</code>\n\n"
        f"👛 <b>Balance:</b> <code>{float(balance):.2f} USDT</code>\n"
        f"{BAR}"
    )


def format_trailing_alert(trade, current_price, trail_percent):
    n = trade.get("trade_no", "?")
    return (
        f"🚀 <b>TRADE #{n} — TRAILING STOP ARMED</b>\n"
        f"{BAR}\n"
        f"🪙 <b>{trade.get('symbol')}</b>   {_dir_badge(trade.get('direction'))}\n"
        f"🎯 <b>TP reached:</b> <code>{float(trade.get('tp', 0)):.6f}</code>\n"
        f"📈 <b>Now:</b> <code>{float(current_price):.6f}</code>\n"
        f"📏 <b>Trail:</b> <code>{trail_percent}%</code>\n"
        f"{BAR}"
    )
