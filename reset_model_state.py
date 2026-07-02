"""
Reset trade history + model state for a fresh training run.

Leaves untouched: paper_balance.json, open_trades.json, cooldowns.json,
signal_hashes.json — none of that is part of model training.

Backs everything up to data/backups/<timestamp>/ before deleting anything,
so this is reversible if you change your mind.

Run this from the same working directory the bot runs from (so the relative
"data" path in trade_manager.py resolves the same way), e.g.:

    railway run python reset_model_state.py

or shell into the Railway container and run it there directly.
"""

import os
import json
import shutil
from datetime import datetime

DATA_DIR = "data"
MODELS_DIR = "/app/data/models"

HISTORY_FILE = os.path.join(DATA_DIR, "trade_history.json")

MODEL_FILES = [
    "xgboost_model.pkl",
    "xgboost_model_challenger.pkl",
    "expected_r_model.pkl",
    "feature_names.pkl",
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

    print(f"\n📦 Backing up to: {backup_dir}\n")

    # 1. Trade history: back up, then reset to empty list (not deleted,
    #    since trade_manager.py expects the file/JSON structure to exist).
    if os.path.exists(HISTORY_FILE):
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(HISTORY_FILE, os.path.join(backup_dir, "trade_history.json"))
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f, indent=4)
        print(f"  ✅ trade history backed up, reset to empty list")
    else:
        print(f"  (skip, not found) {HISTORY_FILE}")

    # 2. Model state: back up and remove entirely (files get recreated on
    #    next successful training run).
    print(f"\n🧹 Clearing model state in {MODELS_DIR}\n")
    for filename in MODEL_FILES:
        backup_and_remove(os.path.join(MODELS_DIR, filename), backup_dir)

    print(f"\n✅ Done. Backup saved at: {backup_dir}")
    print("   Balance, open trades, cooldowns, and signal hashes were left untouched.")
    print("   Model will start training fresh once 30+ new labeled trades are recorded.")


if __name__ == "__main__":
    main()
