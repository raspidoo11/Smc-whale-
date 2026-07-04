"""SQLite persistence backend.

Why blobs, not a wide relational schema: trade records are an ever-growing bag
of strategy/feature fields that change as the model evolves. Pinning them to
rigid columns would mean a migration every time a feature is added. Instead each
trade is stored as a JSON blob in a row, giving us SQLite's atomicity, crash
safety, and WAL concurrency while keeping the flexible dict shape the rest of
the code already speaks. A small KV table holds balance / signal-hashes /
cooldowns.

Public surface mirrors what trade_manager needs:
  read_list(table) / write_list(table, items)   -> open_trades, history
  kv_get(key, default) / kv_set(key, value)     -> balance, signal_hashes, ...

On first use it auto-imports any legacy data/*.json so an existing deployment
migrates seamlessly (the JSON files are left in place as a fallback).
"""

import os
import json
import sqlite3
import threading
import logging

from config import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(DATA_DIR, "smc_whale.db")

_LIST_TABLES = ("open_trades", "history")

_lock = threading.Lock()
_conn = None


def _connect():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS open_trades (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS history (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        _conn.commit()
        _migrate_from_json_if_needed(_conn)
    return _conn


def read_list(table):
    assert table in _LIST_TABLES, f"unknown table {table}"
    conn = _connect()
    with _lock:
        cur = conn.execute(f"SELECT data FROM {table} ORDER BY seq")
        return [json.loads(r[0]) for r in cur.fetchall()]


def write_list(table, items):
    assert table in _LIST_TABLES, f"unknown table {table}"
    conn = _connect()
    payload = [(json.dumps(it, default=str),) for it in items]
    with _lock:
        conn.execute(f"DELETE FROM {table}")
        conn.executemany(f"INSERT INTO {table}(data) VALUES (?)", payload)
        conn.commit()


def kv_get(key, default=None):
    conn = _connect()
    with _lock:
        cur = conn.execute("SELECT value FROM kv WHERE key = ?", (key,))
        row = cur.fetchone()
    return json.loads(row[0]) if row else default


def kv_set(key, value):
    conn = _connect()
    with _lock:
        conn.execute(
            "INSERT INTO kv(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value, default=str)),
        )
        conn.commit()


def _migrate_from_json_if_needed(conn):
    # Only migrate into a fresh DB, so we never clobber live SQLite data.
    open_ct = conn.execute("SELECT COUNT(*) FROM open_trades").fetchone()[0]
    hist_ct = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    kv_ct = conn.execute("SELECT COUNT(*) FROM kv").fetchone()[0]
    if open_ct or hist_ct or kv_ct:
        return

    def _load(name, default):
        path = os.path.join(DATA_DIR, name)
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return default

    ot = _load("open_trades.json", [])
    hist = _load("trade_history.json", [])
    bal = _load("paper_balance.json", None)
    hashes = _load("signal_hashes.json", None)
    cooldowns = _load("cooldowns.json", None)

    migrated = False
    if ot:
        conn.executemany(
            "INSERT INTO open_trades(data) VALUES (?)",
            [(json.dumps(x, default=str),) for x in ot],
        )
        migrated = True
    if hist:
        conn.executemany(
            "INSERT INTO history(data) VALUES (?)",
            [(json.dumps(x, default=str),) for x in hist],
        )
        migrated = True
    for key, val in (("balance", bal), ("signal_hashes", hashes), ("cooldowns", cooldowns)):
        if val is not None:
            conn.execute(
                "INSERT INTO kv(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(val, default=str)),
            )
            migrated = True

    conn.commit()
    if migrated:
        logger.info(f"🗃️  Migrated legacy JSON state into SQLite at {DB_PATH}")
