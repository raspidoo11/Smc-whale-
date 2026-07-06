import asyncio
import os
import schedule
import time
import logging
from datetime import datetime, timezone
from scanner import get_live_symbols as get_top_symbols, get_ohlcv
import strategy
from strategy import get_signal
from paper_trader import calculate_qty
from bybit_executor import execute_trade
from exchange import get_exchange
from telegram_alerts import send_alert
from trade_manager import (
    add_trade,
    trading_allowed,
    trade_exists,
    next_trade_number,
    get_balance,
    reset_daily_pnl,
    get_open_trades,
    get_signal_hash_exists,
    save_signal_hash
)
from trade_monitor import monitor_trades
from xgboost_trainer import train_model_incremental
from trade_manager import get_trade_history
from bybit_executor import EXECUTE_TRADES
from config import (
    MAX_OPEN_TRADES,
    ENTRY_MODE,
    LIMIT_TTL_MINUTES,
    SPREAD_MAX_FRACTION_OF_RISK,
    NEWS_FILTER_ENABLED,
)
from alerts import format_open_alert, format_limit_alert
from risk_manager import can_open_trade, blocked_session_now
from reconcile import reconcile
from market_context import get_market_context
from news_filter import get_news_status

# Wire live market-context enrichment (funding / OI / BTC trend / spread) into
# the signal engine. The backtester never does this, so it stays offline.
strategy.MARKET_CONTEXT_FN = get_market_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

if EXECUTE_TRADES:
    exchange = get_exchange()
else:
    exchange = None

# ==================== CONFIG ====================
# MAX_OPEN_TRADES now comes from config.py (single source of truth).
SCAN_LOCK = asyncio.Lock()           # Prevents overlapping scans


def count_active_trades():
    """OPEN positions + PENDING limit orders both consume a slot — a resting
    order is committed risk the moment it can fill."""
    return len([
        t for t in get_open_trades()
        if t.get("status") in ("OPEN", "PENDING")
    ])


async def get_fresh_symbols(limit=30):
    """Get symbols excluding those we already have open trades or resting
    limit orders on."""
    open_trades = get_open_trades()
    open_symbols = {
        t["symbol"] for t in open_trades
        if t.get("status") in ("OPEN", "PENDING")
    }

    # Get more symbols than needed so we can filter
    all_symbols = get_top_symbols(100)

    fresh_symbols = [s for s in all_symbols if s not in open_symbols][:limit]
    logger.info(f"📊 Fresh symbols selected: {len(fresh_symbols)} (excluded {len(open_symbols)} open/pending)")

    return fresh_symbols


async def scan():
    """Main scanning function with lock + fresh symbols"""
    async with SCAN_LOCK:   # Prevent overlapping scans
        await monitor_trades()

        try:
            logger.info("🔍 Starting signal scan...")

            if not trading_allowed():
                logger.info("⛔ Daily loss limit reached. Trading paused.")
                return

            if NEWS_FILTER_ENABLED:
                paused, news_reason = get_news_status()
                if paused:
                    logger.info(f"📰 Entries paused: {news_reason}")
                    return

            blocked = blocked_session_now()
            if blocked:
                logger.info(f"🌙 {blocked} session is blocked (BLOCKED_SESSIONS) — no new entries")
                return

            if count_active_trades() >= MAX_OPEN_TRADES:
                logger.info(f"📉 Max open+pending trades ({MAX_OPEN_TRADES}) reached. Skipping scan.")
                return

            symbols = await get_fresh_symbols(30)

            results = []

            for symbol in symbols:
                try:
                    df_15m = get_ohlcv(symbol, "15m", 200)
                    df_5m = get_ohlcv(symbol, "5m", 200)

                    if df_15m is None or df_5m is None:
                        continue

                    signal = get_signal(symbol, df_15m, df_5m)

                    if signal:
                        # (strategy already logged the compact 🎯 line)
                        # Check for duplicate using signal_hash
                        if get_signal_hash_exists(signal.get("signal_hash")):
                            logger.info(f"⏭️ Duplicate signal_hash skipped: {symbol}")
                            continue

                        # Spread gate: if the bid-ask spread eats too much of
                        # the planned risk, the edge is gone before we start.
                        spread_pct = signal.get("spread_pct")
                        if spread_pct is not None:
                            spread_abs = spread_pct / 100 * signal["entry"]
                            max_spread = SPREAD_MAX_FRACTION_OF_RISK * abs(signal["entry"] - signal["sl"])
                            if spread_abs > max_spread:
                                logger.info(
                                    f"🕳️ {symbol}: spread {spread_pct:.3f}% eats "
                                    f">{SPREAD_MAX_FRACTION_OF_RISK:.0%} of risk — skipping"
                                )
                                continue

                        # Limit mode: enter at the retracement level, resize
                        # SL/TP economics around the actual limit entry.
                        if ENTRY_MODE == "limit":
                            limit_entry = float(signal["limit_price"])
                            rr = float(signal.get("rr_multiplier", 1.5))
                            sl = float(signal["sl"])
                            if signal["direction"] == "LONG":
                                tp = limit_entry + (limit_entry - sl) * rr
                            else:
                                tp = limit_entry - (sl - limit_entry) * rr
                            signal["signal_close"] = signal["entry"]  # keep original for reference
                            signal["entry"] = limit_entry
                            signal["tp"] = float(tp)
                            signal["entry_type"] = "limit"
                        else:
                            signal["entry_type"] = "market"

                        qty = calculate_qty(signal["entry"], signal["sl"])
                        signal["qty"] = qty
                        results.append({"symbol": symbol, **signal})

                except Exception as e:
                    logger.exception(f"Symbol scan error: {symbol} | {e}")

            if not results:
                logger.info("No valid signals this scan.")
                return

            # Rank by confidence (session bonus already applied in strategy)
            results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            top_signals = results[:3]

            for trade in top_signals:
                if count_active_trades() >= MAX_OPEN_TRADES:
                    break

                trade_no = next_trade_number()
                balance = get_balance()["balance"]

                is_limit = trade.get("entry_type") == "limit"

                # FIX: previously this rebuilt a trimmed whitelist dict here,
                # which silently dropped sweep/fvg/volume_spike/displacement,
                # atr/body/volume/volume_ma, hour/day_of_week, market_regime,
                # atr_percentile, and ai_prob before they ever reached
                # trade_history.json. That's why the trainer's feature
                # importance report showed those features at 0.000 — they
                # were never persisted, not "learned and ignored." Spreading
                # `trade` (symbol + full signal dict + qty) preserves
                # everything strategy.py computed, then we layer on the
                # execution-specific fields on top.
                trade_data = {
                    **trade,
                    "entry": float(trade["entry"]),
                    "sl": float(trade["sl"]),
                    "tp": float(trade["tp"]),
                    "qty": float(trade["qty"]),
                    # Limit orders rest as PENDING until the monitor confirms
                    # the fill (or expires them); market entries open now.
                    "status": "PENDING" if is_limit else "OPEN",
                    "trade_no": trade_no,
                    "placed_at": datetime.now(timezone.utc).isoformat(),
                }

                # Portfolio-level risk gate: even if we're under MAX_OPEN_TRADES,
                # reject setups that would over-concentrate direction/alts or
                # push total open risk past the configured heat cap.
                allowed, reason = can_open_trade(trade_data, get_open_trades(), balance)
                if not allowed:
                    logger.info(f"🚦 Skipping {trade['symbol']}: {reason}")
                    continue

                if EXECUTE_TRADES:
                    order = await execute_trade(trade_data)

                    if not order:
                        logger.error(f"❌ Failed to execute: {trade['symbol']}")
                        continue

                    # Keep the exchange order id so the monitor can track the
                    # resting limit order's fill/cancel state.
                    order_id = (order.get("result") or {}).get("orderId")
                    if order_id:
                        trade_data["order_id"] = order_id

                    logger.info(f"✅ Live {'limit order placed' if is_limit else 'trade executed'}: {trade['symbol']}")
                else:
                    logger.info(f"📝 Paper {'limit order placed' if is_limit else 'trade opened'}: {trade['symbol']}")

                add_trade(trade_data)

                # Save signal_hash to prevent duplicates
                if trade.get("signal_hash"):
                    save_signal_hash(trade["signal_hash"])

                if is_limit:
                    await send_alert(format_limit_alert(trade_data, LIMIT_TTL_MINUTES))
                else:
                    await send_alert(format_open_alert(trade_data))

        except Exception as e:
            logger.exception(f"SCAN FAILED: {e}")

        # NOTE: model retraining used to live here, at the end of scan()'s
        # try block. That was broken two ways in a row:
        #   1. It was the last line inside the try, so any exception earlier
        #      in the scan (execute_trade, send_alert, etc.) skipped it via
        #      "SCAN FAILED" — but you confirmed SCAN FAILED never appears.
        #   2. More importantly: scan() has THREE early `return` statements
        #      above (daily loss limit reached / max open trades reached /
        #      "no valid signals this scan") that exit the whole coroutine,
        #      not just the try block. "No valid signals" is the common
        #      case, so training was being skipped on most cycles even
        #      without any error at all — silently, by design, no log line.
        # Retraining is now its own independent scheduled job (see
        # run_retrain_sync below + the schedule.every(...) line in main()),
        # decoupled entirely from whether scan() found signals, executed
        # trades, or hit an error.


# ==================== SCHEDULER WRAPPERS ====================

async def run_monitor():
    try:
        await monitor_trades()
    except Exception as e:
        logger.exception(f"Monitor error: {e}")


def retrain_model():
    """Independent retrain job — deliberately NOT called from inside scan(),
    since scan() has multiple early `return`s (no signals / max trades /
    daily limit hit) that would skip it silently. Runs on its own schedule
    instead, so it always gets a chance to fire regardless of scan outcome."""
    try:
        trade_count = len(get_trade_history())
        if trade_count >= 10:
            logger.info(f"🧠 Triggering train_model_incremental() (trade_count={trade_count})")
            train_model_incremental()
        else:
            logger.info(f"🧠 Skipping retrain, not enough trades yet ({trade_count}/10)")
    except Exception as e:
        logger.exception(f"MODEL RETRAIN FAILED: {e}")


# Tracks which 5m candle we last scanned, so scans fire once per candle CLOSE
# (within ~20s of it) instead of every 5 minutes from whenever the process
# happened to boot. With the forming candle dropped in the scanner, this means
# each scan evaluates the freshly-closed candle almost immediately, instead of
# up to 5 minutes late.
_last_scanned_bucket = None


def _current_candle_bucket():
    now = datetime.now(timezone.utc)
    return now.replace(minute=now.minute - now.minute % 5, second=0, microsecond=0)


def run_scan_sync(force=False):
    global _last_scanned_bucket
    try:
        bucket = _current_candle_bucket()
        if not force and bucket == _last_scanned_bucket:
            return  # this candle was already scanned
        _last_scanned_bucket = bucket
        asyncio.run(scan())
    except Exception as e:
        logger.exception(f"Scan wrapper error: {e}")


def run_monitor_sync():
    try:
        asyncio.run(run_monitor())
    except Exception as e:
        logger.exception(f"Monitor wrapper error: {e}")


def run_reconcile_sync():
    """Live/demo only: sync local open trades + balance with the exchange.
    No-op in paper mode (reconcile() checks EXECUTE_TRADES internally)."""
    try:
        asyncio.run(reconcile())
    except Exception as e:
        logger.exception(f"Reconcile wrapper error: {e}")


def heartbeat():
    logger.info("💚 Worker alive")


def daily_reset():
    reset_daily_pnl()
    asyncio.run(send_alert("📅 New trading day started. Daily PnL reset."))


async def startup():
    await send_alert("🚀 <b>SMC Whale Bot Started</b>\nPaper Mode + Session Bonus Active")


def maybe_import_pretrained():
    """Install a committed ./pretrained/ model into MODELS_DIR on boot.

    The Railway volume mounts over /app/data, shadowing any model files
    committed to the repo — so pretrain.py exports its artifacts to
    ./pretrained/ (versioned in git) and this copies them onto the volume.
    Marker-guarded by the pretrained build's trained_at timestamp: each
    pretrained build imports exactly ONCE, so live fine-tuning afterwards is
    never clobbered by a restart. Set PRETRAINED_IMPORT=false to disable."""
    if os.getenv("PRETRAINED_IMPORT", "true").lower() != "true":
        return
    try:
        import json
        import shutil
        from config import MODELS_DIR

        meta_path = os.path.join("pretrained", "training_metadata.json")
        if not os.path.exists(meta_path):
            return
        with open(meta_path) as f:
            meta = json.load(f)
        build_id = meta.get("trained_at", "")

        marker_path = os.path.join(MODELS_DIR, "pretrained_import_marker.json")
        if os.path.exists(marker_path):
            with open(marker_path) as f:
                if json.load(f).get("imported_build") == build_id:
                    return  # this build already imported; don't clobber live fine-tuning

        for name in os.listdir("pretrained"):
            shutil.copy2(os.path.join("pretrained", name), os.path.join(MODELS_DIR, name))
        with open(marker_path, "w") as f:
            json.dump({"imported_build": build_id}, f)

        logger.info(
            f"🧠 Imported pretrained base model (built {build_id}, "
            f"CV AUC {meta.get('cv_mean_auc', '?')}, "
            f"{meta.get('corpus_trades', '?')} corpus trades)"
        )
    except Exception as e:
        logger.exception(f"Pretrained import failed (bot continues normally): {e}")


def maybe_backfill_on_start():
    """Set BACKFILL_ON_START=true in Railway Variables to seed trade history
    with backtest-simulated training rows on the next boot — no CLI needed.
    Idempotent (skips if backfilled rows already exist), so it's safe to leave
    the variable set. Any failure is logged and never blocks the bot."""
    if os.getenv("BACKFILL_ON_START", "false").lower() != "true":
        return
    try:
        from backfill_from_backtest import run_backfill
        logger.info("🧪 BACKFILL_ON_START=true — running backtest backfill (one-time, ~1-2 min)...")
        # The signal engine logs one line per evaluated bar — over a ~24k-bar
        # backfill that alone exceeds Railway's 500 logs/sec limit. Silence it
        # for the duration, exactly like the CLI entry point does.
        strat_logger = logging.getLogger("strategy")
        prev_level = strat_logger.level
        strat_logger.setLevel(logging.WARNING)
        try:
            summary = run_backfill()
        finally:
            strat_logger.setLevel(prev_level)
        logger.info(f"🧪 Backfill result: {summary}")
    except Exception as e:
        logger.exception(f"Backfill on start failed (bot continues normally): {e}")


def main():
    logger.info("🚀 Starting SMC Whale AI (Improved Version)")
    logger.info(f"📊 Max Open Trades: {MAX_OPEN_TRADES}")
    logger.info("🔒 Using asyncio.Lock() for scan protection")
    logger.info("🕒 Session bonus system enabled")

    asyncio.run(startup())
    maybe_import_pretrained()
    maybe_backfill_on_start()
    run_scan_sync(force=True)
    run_monitor_sync()
    retrain_model()

    schedule.every(1).minutes.do(heartbeat)
    schedule.every(35).seconds.do(run_monitor_sync)
    # Checked every 20s but only fires once per NEW closed 5m candle — i.e.
    # candle-aligned scanning with <=20s latency after each close.
    schedule.every(20).seconds.do(run_scan_sync)
    schedule.every(10).minutes.do(retrain_model)
    schedule.every(2).minutes.do(run_reconcile_sync)
    schedule.every().day.at("00:00").do(daily_reset)

    logger.info("✅ Scheduler started")

    while True:
        try:
            schedule.run_pending()
            time.sleep(5)
        except Exception as e:
            logger.exception(f"Main loop error: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
