"""SQLite backend: blob roundtrip, KV, JSON->SQLite migration, and the
trade_manager dispatch path end to end."""

import json
import os

import pytest

import db as db_module
import trade_manager as tm


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Point the DB at a temp dir and reset the module-level connection so each
    test gets an isolated database."""
    monkeypatch.setattr(db_module, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(db_module, "_conn", None)
    yield tmp_path
    if db_module._conn is not None:
        db_module._conn.close()
        db_module._conn = None


def test_list_roundtrip(fresh_db):
    items = [{"symbol": "BTCUSDT", "status": "OPEN"}, {"symbol": "ETHUSDT", "status": "OPEN"}]
    db_module.write_list("open_trades", items)
    assert db_module.read_list("open_trades") == items
    # write_list replaces, not appends
    db_module.write_list("open_trades", items[:1])
    assert db_module.read_list("open_trades") == items[:1]


def test_kv_roundtrip(fresh_db):
    assert db_module.kv_get("balance", {"balance": 1}) == {"balance": 1}
    db_module.kv_set("balance", {"balance": 42.5, "daily_pnl": -1})
    assert db_module.kv_get("balance")["balance"] == 42.5


def test_migration_from_json(fresh_db):
    tmp = fresh_db
    with open(tmp / "open_trades.json", "w") as f:
        json.dump([{"symbol": "BTCUSDT", "status": "OPEN"}], f)
    with open(tmp / "trade_history.json", "w") as f:
        json.dump([{"symbol": "ETHUSDT", "status": "WIN", "pnl": 3.0}], f)
    with open(tmp / "paper_balance.json", "w") as f:
        json.dump({"balance": 123.0, "daily_pnl": 0.0}, f)

    # First connect triggers migration.
    db_module._connect()
    assert db_module.read_list("open_trades")[0]["symbol"] == "BTCUSDT"
    assert db_module.read_list("history")[0]["status"] == "WIN"
    assert db_module.kv_get("balance")["balance"] == 123.0


def test_trade_manager_sqlite_dispatch(fresh_db, monkeypatch):
    monkeypatch.setattr(tm, "STORAGE_BACKEND", "sqlite")
    # start clean
    tm.save_open_trades([])
    tm.save_trade_history([])

    tm.add_trade({"symbol": "BTCUSDT", "status": "OPEN", "entry": 100.0, "sl": 98.0,
                  "direction": "LONG", "qty": 1.0, "trade_no": 1})
    assert len(tm.get_open_trades()) == 1

    closed = tm.close_trade("BTCUSDT", 104.0, "WIN", extra_fields={"pnl": 3.9})
    assert closed["pnl"] == 3.9
    assert tm.get_open_trades() == []
    hist = tm.get_trade_history()
    assert hist[0]["status"] == "WIN" and hist[0]["pnl"] == 3.9
