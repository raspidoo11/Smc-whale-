"""Paper balance must move when a trailing stop closes a winner."""

import asyncio
import os
import tempfile
import importlib
import sys
import types


def _fresh_stack(tmpdir):
    os.environ["DATA_DIR"] = tmpdir
    os.environ["STORAGE_BACKEND"] = "sqlite"
    os.environ["EXECUTE_TRADES"] = "false"

    stub = types.ModuleType("exchange")
    stub._TEST_STUB = True

    class FakeEx:
        def __init__(self, last):
            self._last = last

        def fetch_ticker(self, symbol):
            return {"last": self._last}

    stub.FakeEx = FakeEx
    stub.get_exchange = lambda: FakeEx(109.40)
    stub.get_trade_client = lambda: None
    sys.modules["exchange"] = stub

    import config
    importlib.reload(config)
    config.DATA_DIR = tmpdir
    config.MODELS_DIR = os.path.join(tmpdir, "models")

    import db
    importlib.reload(db)

    import trade_manager as tm
    importlib.reload(tm)

    import paper_trader as pt
    importlib.reload(pt)

    import bybit_executor as be
    importlib.reload(be)
    be.EXECUTE_TRADES = False

    import trade_monitor as mon
    importlib.reload(mon)
    mon.EXECUTE_TRADES = False
    mon.exchange = FakeEx(109.40)

    async def _no_alert(*a, **k):
        return None

    mon.send_alert = _no_alert
    return tm, pt, mon, FakeEx


def test_paper_trail_close_credits_balance_and_mirrors_json():
    tmpdir = tempfile.mkdtemp()
    tm, pt, mon, FakeEx = _fresh_stack(tmpdir)

    tm.save_balance({"balance": 100.0, "daily_pnl": 0.0})
    trade = {
        "symbol": "SOL/USDT:USDT",
        "direction": "LONG",
        "entry": 100.0,
        "sl": 99.0,
        "tp": 103.0,
        "qty": 5.0,
        "status": "OPEN",
        "entry_fee": 0.02,
        "entry_fee_applied": True,
        "trailing_stop_active": True,
        "trail_anchor": 110.0,
        "trail_percent": 0.5,
        "trade_no": 42,
    }
    tm.add_trade(trade)

    # Price below the trail stop from a 110 peak.
    mon.exchange = FakeEx(109.40)
    asyncio.run(mon.monitor_trades())

    bal = tm.get_balance()
    assert bal["balance"] > 100.0, bal
    assert bal["daily_pnl"] > 0.0, bal
    assert tm.get_open_trades() == []

    hist = tm.get_trade_history()
    assert len(hist) == 1
    assert hist[0]["exit_reason"] == "Trailing Stop Hit"
    assert hist[0]["status"] == "WIN"
    assert hist[0]["pnl"] > 0

    # Legacy file mirror must stay in sync (operators peek at this file).
    import json
    with open(os.path.join(tmpdir, "paper_balance.json")) as f:
        mirrored = json.load(f)
    assert abs(mirrored["balance"] - bal["balance"]) < 1e-9


def test_paper_trail_tight_scalp_still_credits_nonzero():
    """Tight arm zone used to pin exit at BE (~$0). Profit lock must pay."""
    tmpdir = tempfile.mkdtemp()
    tm, pt, mon, FakeEx = _fresh_stack(tmpdir)

    tm.save_balance({"balance": 100.0, "daily_pnl": 0.0})
    trade = {
        "symbol": "ETH/USDT:USDT",
        "direction": "LONG",
        "entry": 100.0,
        "sl": 99.8,
        "tp": 100.3,
        "qty": 10.0,
        "status": "OPEN",
        "entry_fee_applied": True,
        "entry_fee": 0.04,
        "trailing_stop_active": True,
        "trail_anchor": 100.27,
        "trail_percent": 0.3,
        "trade_no": 1,
    }
    tm.add_trade(trade)

    # Breach the profit-locked stop (above entry, not pure BE).
    mon.exchange = FakeEx(100.10)
    asyncio.run(mon.monitor_trades())

    bal = tm.get_balance()
    assert bal["balance"] > 100.0, bal
    hist = tm.get_trade_history()
    assert hist and hist[0]["pnl"] > 0
