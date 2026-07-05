"""Offline pretraining: build a multi-regime training corpus and a base model.

Replays the live signal engine over months of history across a basket of
symbols (with REAL historical funding / OI / BTC-trend / Fear&Greed via
historical_context.py), evaluates honestly with PURGED WALK-FORWARD
cross-validation, and — only with --promote — installs the result as the base
champion. Live retraining then keeps fine-tuning: a live-window challenger
must beat this champion on holdout AUC to replace it (existing promotion
logic), and strategy's trust ramp still keys off REAL closed trades only.

Purged walk-forward: each fold trains ONLY on trades that fully exited before
the test block starts, minus an embargo gap — no shuffled-time leakage, no
training on a trade whose life overlaps the test period. This is the
evaluation an honest "is there edge?" answer requires; shuffled CV flatters.

Usage:
    python pretrain.py --days 90                     # evaluate only (default)
    python pretrain.py --days 180 --promote          # install as champion
    python pretrain.py --symbols "BTC/USDT:USDT" "ETH/USDT:USDT" --days 30

Per-symbol results are cached in data/pretrain_cache/ so re-runs (and crashes)
don't refetch/replay finished symbols. Delete the cache to force a rebuild.
"""

import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import DATA_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(DATA_DIR, "pretrain_cache")

# Bump when the SIGNAL SNAPSHOT schema changes (new persisted fields): cached
# corpus rows generated before the change would silently miss the new feature
# (defaulting to 0 everywhere = a dead constant in CV). v2 = added CRT.
CACHE_VERSION = 2

# Liquid, non-meme perps across sectors — regime + symbol diversity.
DEFAULT_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "ADA/USDT:USDT", "LINK/USDT:USDT", "AVAX/USDT:USDT",
    "DOT/USDT:USDT", "LTC/USDT:USDT", "ATOM/USDT:USDT", "NEAR/USDT:USDT",
    "APT/USDT:USDT", "ARB/USDT:USDT", "OP/USDT:USDT", "SUI/USDT:USDT",
    "TON/USDT:USDT", "TRX/USDT:USDT", "UNI/USDT:USDT", "AAVE/USDT:USDT",
]

BARS_PER_DAY = 288  # 5m bars


def _exit_time_ms(trade):
    """Approximate exit timestamp: entry_time + bars_held x 5m."""
    t = pd.to_datetime(trade.get("entry_time"), utc=True)
    bars = int(trade.get("bars_held", 1) or 1)
    return int(t.timestamp() * 1000) + bars * 5 * 60 * 1000


def _entry_time_ms(trade):
    return int(pd.to_datetime(trade.get("entry_time"), utc=True).timestamp() * 1000)


def purged_walk_forward_folds(entry_ms, exit_ms, n_folds=5, embargo_ms=24 * 3600 * 1000):
    """Expanding-window walk-forward with purge+embargo over chronologically
    sorted rows. Yields (train_idx, test_idx); train rows must have EXITED at
    least `embargo_ms` before the first test entry. First block is train-only.
    """
    n = len(entry_ms)
    bounds = [int(n * i / (n_folds + 1)) for i in range(1, n_folds + 2)]
    prev = bounds[0]
    for b in bounds[1:]:
        test_idx = list(range(prev, b))
        if not test_idx:
            prev = b
            continue
        test_start = entry_ms[test_idx[0]]
        train_idx = [i for i in range(prev) if exit_ms[i] <= test_start - embargo_ms]
        if train_idx and len(set(train_idx) & set(test_idx)) == 0:
            yield train_idx, test_idx
        prev = b


def collect_symbol(sym, days, provider=None, cache_dir=CACHE_DIR):
    """Fetch + replay one symbol (or load from cache). Returns closed trades.
    `provider` is a shared HistoricalContextProvider — global series (BTC
    trend, Fear&Greed) are preloaded once for the whole run, per-symbol series
    (funding, OI) here."""
    os.makedirs(cache_dir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", sym)
    cache_file = os.path.join(cache_dir, f"{safe}_{days}d_v{CACHE_VERSION}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            trades = json.load(f)
        logger.info(f"   {sym}: {len(trades)} trades (cached)")
        return trades

    from backtester import simulate, fetch_ohlcv_paginated
    from historical_context import HistoricalContextProvider

    candles = days * BARS_PER_DAY
    df5 = fetch_ohlcv_paginated(sym, "5m", candles)
    df15 = fetch_ohlcv_paginated(sym, "15m", max(candles // 3, 200))
    if len(df5) < 500:
        logger.warning(f"   {sym}: only {len(df5)} candles — skipping")
        return []

    start_ms = int(df5.index[0].timestamp() * 1000)
    end_ms = int(df5.index[-1].timestamp() * 1000)

    if provider is None:
        provider = HistoricalContextProvider()
    if provider._btc is None and provider._fng is None:
        provider.preload_global(start_ms, end_ms)
    provider.preload(sym, start_ms, end_ms)

    trades, metrics = simulate(sym, df5, df15, use_xgboost=False,
                               context_provider=provider)
    for t in trades:
        t["source"] = "backtest"

    with open(cache_file, "w") as f:
        json.dump(trades, f, default=str)
    logger.info(f"   {sym}: {len(trades)} trades (win rate {metrics.get('win_rate', 0):.0%})")
    return trades


def run_pretrain(symbols, days, n_folds, embargo_hours, promote):
    from xgboost_trainer import (
        build_feature_frame, prepare_X_y, fit_candidate_model,
        fit_expected_r_model, evaluate_model, _dump_and_verify,
        MODEL_PATH, FEATURE_PATH, EXPECTED_R_MODEL_PATH,
        EXPECTED_R_FEATURE_PATH, METADATA_PATH, save_json,
        MIN_TRADES_FOR_ENSEMBLE,
    )

    # ---- 1. Corpus ----
    from historical_context import HistoricalContextProvider
    shared_provider = HistoricalContextProvider()

    corpus = []
    for sym in symbols:
        try:
            corpus += collect_symbol(sym, days, provider=shared_provider)
        except Exception as e:
            logger.warning(f"   {sym} failed ({e}) — skipping")

    corpus.sort(key=lambda t: t.get("entry_time", ""))
    logger.info(f"Corpus: {len(corpus)} trades across {len(symbols)} symbols, {days} days")
    if len(corpus) < 100:
        print(f"Corpus too small ({len(corpus)} trades) — need >=100. More days/symbols.")
        return None

    df = build_feature_frame(corpus)
    entry_ms = [_entry_time_ms(t) for t in corpus]
    exit_ms = [_exit_time_ms(t) for t in corpus]
    pos_rate = df["target"].mean()
    logger.info(f"Labels: {int(df['target'].sum())} meaningful wins / {len(df)} "
                f"({pos_rate:.0%} positive at R>=label threshold)")

    # ---- 2. Purged walk-forward CV ----
    embargo_ms = int(embargo_hours * 3600 * 1000)
    aucs, briers = [], []
    use_ensemble = len(df) >= MIN_TRADES_FOR_ENSEMBLE
    for k, (tr_idx, te_idx) in enumerate(
        purged_walk_forward_folds(entry_ms, exit_ms, n_folds, embargo_ms), 1
    ):
        tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
        if tr_df["target"].nunique() < 2 or te_df["target"].nunique() < 2:
            logger.info(f"   fold {k}: single-class, skipped")
            continue
        X_tr, y_tr = prepare_X_y(tr_df)
        X_te, y_te = prepare_X_y(te_df)
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
        # Uniform weights: recency decay (built for the live 220-trade window)
        # would collapse a multi-regime corpus to ~87 effective samples and
        # produce degenerate constant models on large folds.
        model, _, _ = fit_candidate_model(X_tr, y_tr, use_ensemble=use_ensemble,
                                          recency_half_life=None)
        m = evaluate_model(model, X_te, y_te) or {}
        if m.get("auc") is not None:
            aucs.append(m["auc"])
            briers.append(m.get("brier"))
        logger.info(f"   fold {k}: train {len(tr_idx)} / test {len(te_idx)} "
                    f"| AUC {m.get('auc')} | Brier {m.get('brier')}")

    if not aucs:
        print("No evaluable folds — corpus too small or single-class.")
        return None

    mean_auc = float(np.mean(aucs))
    mean_brier = float(np.mean([b for b in briers if b is not None]))
    print(f"\nPurged walk-forward: mean AUC {mean_auc:.3f} "
          f"(folds: {', '.join(f'{a:.3f}' for a in aucs)}) | mean Brier {mean_brier:.3f}")
    print("Honest read: >0.55 = real signal; ~0.50 = no edge learned yet.")

    # ---- 3. Final fit + optional promotion ----
    if not promote:
        print("\nEvaluation-only run (no model written). Re-run with --promote to install.")
        return mean_auc

    X_full, y_full = prepare_X_y(df)
    final_model, _, model_type = fit_candidate_model(X_full, y_full, use_ensemble=use_ensemble,
                                                     recency_half_life=None)
    _dump_and_verify(final_model, MODEL_PATH, "PRETRAINED base model")
    _dump_and_verify(X_full.columns.tolist(), FEATURE_PATH, "feature name list")

    r_target = df["realized_r"]
    r_model = fit_expected_r_model(X_full, r_target)
    _dump_and_verify(r_model, EXPECTED_R_MODEL_PATH, "expected-R model")
    _dump_and_verify(X_full.columns.tolist(), EXPECTED_R_FEATURE_PATH, "expected-R feature list")

    save_json(METADATA_PATH, {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "pretrained_base": True,
        "model_type": model_type,
        "corpus_trades": len(df),
        "corpus_symbols": symbols,
        "corpus_days": days,
        "cv_mean_auc": mean_auc,
        "cv_fold_aucs": aucs,
        "cv_mean_brier": mean_brier,
        "n_features": len(X_full.columns),
    })

    # Also export the artifacts into ./pretrained/ (committed to git). The
    # Railway volume mounts over /app/data, shadowing anything committed
    # there — main.maybe_import_pretrained() copies these into MODELS_DIR on
    # boot instead, marker-guarded so it happens once per pretrained build.
    import shutil
    os.makedirs("pretrained", exist_ok=True)
    for src in (MODEL_PATH, FEATURE_PATH, EXPECTED_R_MODEL_PATH,
                EXPECTED_R_FEATURE_PATH, METADATA_PATH):
        shutil.copy2(src, os.path.join("pretrained", os.path.basename(src)))
    print(f"\n[OK] Pretrained base installed as champion ({model_type}, "
          f"{len(df)} trades, CV AUC {mean_auc:.3f}).")
    print("Artifacts exported to ./pretrained/ — commit them and Railway will "
          "import the model on next boot.")
    print("Live retraining continues on top — a live challenger must beat this "
          "champion on holdout AUC to replace it.")
    return mean_auc


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("strategy").setLevel(logging.WARNING)

    p = argparse.ArgumentParser(description="Pretrain a base model on a multi-regime backtest corpus.")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--embargo-hours", type=float, default=24)
    p.add_argument("--promote", action="store_true",
                   help="Install the pretrained model as champion (default: evaluate only)")
    args = p.parse_args()

    run_pretrain(args.symbols, args.days, args.folds, args.embargo_hours, args.promote)


if __name__ == "__main__":
    main()
