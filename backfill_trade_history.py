"""
One-off backfill for trade_history.json entries written before the
paper_trader.py / trade_manager.py fixes.

Two bugs corrupted every closed trade going forward until the patch:
  1. status was always "LOSS" (compared exit_reason against a dead string
     that no longer matches any real exit path).
  2. pnl (and exit_fee) were set on the trade dict AFTER it was already
     saved to trade_history.json, so they never actually persisted -- pnl
     shows as None for every trade.

This script recomputes pnl/exit_fee/status from entry, exit_price, qty,
and direction (the same math close_paper_trade_with_fees uses) and rewrites
trade_history.json in place. It does NOT touch paper_balance.json --
update_balance() was always called with the correctly-computed pnl at the
time each trade closed, so the balance itself was never wrong, only the
history record of it.

Usage:
    python3 backfill_trade_history.py            # dry run, prints a report, writes nothing
    python3 backfill_trade_history.py --apply     # writes the corrected file (after a backup)
"""
import json
import os
import sys
import shutil
from datetime import datetime, timezone

DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "trade_history.json")
FEE_RATE = 0.0004


def load_history():
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


def backup_history():
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = f"{HISTORY_FILE}.backup.{ts}"
    shutil.copy2(HISTORY_FILE, backup_path)
    return backup_path


def recompute_trade(trade):
    """Returns (updated_trade, changed: bool, reason: str|None) -- reason is
    set if the trade couldn't be recomputed (missing required fields)."""
    entry = trade.get("entry")
    exit_price = trade.get("exit_price")
    qty = trade.get("qty")
    direction = trade.get("direction")

    if entry is None or exit_price is None or qty is None or direction not in ("LONG", "SHORT"):
        return trade, False, "missing entry/exit_price/qty/direction"

    entry = float(entry)
    exit_price = float(exit_price)
    qty = float(qty)

    if direction == "LONG":
        raw_pnl = (exit_price - entry) * qty
    else:
        raw_pnl = (entry - exit_price) * qty

    exit_fee = round(exit_price * qty * FEE_RATE, 2)
    pnl_after_fees = round(raw_pnl - exit_fee, 2)
    correct_status = "WIN" if pnl_after_fees > 0 else "LOSS"

    old_status = trade.get("status")
    old_pnl = trade.get("pnl")

    changed = (old_status != correct_status) or (old_pnl != pnl_after_fees)

    trade["pnl"] = pnl_after_fees
    trade["exit_fee"] = exit_fee
    trade["status"] = correct_status
    # entry_fee should already be present from when the trade was opened
    # (open_paper_trade sets it at creation time, before any of the closing
    # bugs applied) -- only fill it in if it's genuinely missing.
    if trade.get("entry_fee") is None:
        trade["entry_fee"] = round(entry * qty * FEE_RATE, 2)

    return trade, changed, None


def main():
    apply_changes = "--apply" in sys.argv

    if not os.path.exists(HISTORY_FILE):
        print(f"❌ {HISTORY_FILE} not found. Are you running this from /app?")
        return

    history = load_history()
    print(f"Loaded {len(history)} trades from {HISTORY_FILE}\n")

    win_before = sum(1 for t in history if t.get("status") == "WIN")
    loss_before = sum(1 for t in history if t.get("status") == "LOSS")
    print(f"BEFORE: WIN={win_before}  LOSS={loss_before}")

    updated = []
    changed_count = 0
    skipped = []

    for trade in history:
        new_trade, changed, skip_reason = recompute_trade(dict(trade))  # work on a copy
        if skip_reason:
            skipped.append((trade.get("trade_no"), trade.get("symbol"), skip_reason))
            updated.append(trade)  # leave untouched
            continue
        if changed:
            changed_count += 1
        updated.append(new_trade)

    win_after = sum(1 for t in updated if t.get("status") == "WIN")
    loss_after = sum(1 for t in updated if t.get("status") == "LOSS")
    print(f"AFTER:  WIN={win_after}  LOSS={loss_after}")
    print(f"\n{changed_count} trade(s) corrected.")

    if skipped:
        print(f"\n⚠️  {len(skipped)} trade(s) skipped (missing required fields):")
        for trade_no, symbol, reason in skipped:
            print(f"   trade_no={trade_no} symbol={symbol} -> {reason}")

    print("\nSample of corrected trades (first 10):")
    for t in updated[:10]:
        print(f"   #{t.get('trade_no')} {t.get('symbol')} {t.get('status')} pnl=${t.get('pnl')}")

    if not apply_changes:
        print("\n🔍 DRY RUN ONLY -- nothing was written.")
        print("   Re-run with --apply to write the corrected file:")
        print("   python3 backfill_trade_history.py --apply")
        return

    backup_path = backup_history()
    print(f"\n💾 Backed up original to: {backup_path}")

    save_history(updated)
    print(f"✅ Wrote corrected history to {HISTORY_FILE}")


if __name__ == "__main__":
    main()
