"""
Reset trade history + model state for a fresh training run.

Backend-aware: trade history is cleared through trade_manager, so this works
whether STORAGE_BACKEND is "sqlite" (default) or "json". Model artifacts are
removed from config.MODELS_DIR.

Leaves untouched: balance, open trades, cooldowns, signal hashes — none of
that is part of model training.

Everything is backed up to data/backups/<timestamp>/ first, so it's reversible.

Run from the same working directory the bot runs from, e.g.:

    railway run python reset_model_state.py

or shell into the Railway container and run it there directly.
"""

import os
import json
import shutil
from datetime import datetime

from config import DATA_DIR, MODELS_DIR
import trade_manager

MODEL_FILES = [
    "xgboost_model.pkl",
    "xgboost_model_challenger.pkl",
    "expected_r_model.pkl",
    "feature_names.pkl",
    "expected_r_feature_names.pkl",
    "feature_importance_history.json",
    "training_metrics_history.json",
    "training_metadata.json",
    "diagnostics_report.json",
    "diagnostics_state.json",
]


def backup_and_remove(path, backup_dir):
    if not os.path.exists(path):
        print(f"  (skip, not found) {path}")
        return
    os.makedirs(backup_dir, exist_ok=True)
    shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))
    os.remove(path)
    print(f"  ✅ backed up + removed: {path}")


def main():
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(DATA_DIR, "backups", timestamp)
    os.makedirs(backup_dir, exist_ok=True)

    print(f"\n📦 Backing up to: {backup_dir}")
    print(f"   Storage backend: {trade_manager.STORAGE_BACKEND}\n")

    # 1. Trade history — read via the active backend, back it up, then clear.
    history = trade_manager.get_trade_history()
    with open(os.path.join(backup_dir, "trade_history.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)
    trade_manager.save_trade_history([])
    print(f"  ✅ backed up {len(history)} trades, history cleared")

    # 2. Model state: back up and remove (recreated on next training run).
    print(f"\n🧹 Clearing model state in {MODELS_DIR}\n")
    for filename in MODEL_FILES:
        backup_and_remove(os.path.join(MODELS_DIR, filename), backup_dir)

    print(f"\n✅ Done. Backup saved at: {backup_dir}")
    print("   Balance, open trades, cooldowns, and signal hashes were left untouched.")
    print("   Model will start training fresh once 30+ new labeled trades are recorded.")


if __name__ == "__main__":
    main()
