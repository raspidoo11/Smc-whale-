import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# Storage paths (single source of truth)
# ==========================================================
# Previously model paths were hardcoded to "/app/data/models" inside
# xgboost_trainer.py (Railway-only, breaks locally on Windows/macOS) while
# trade_manager.py used a relative "data/". Both now derive from here so the
# same code runs locally and in production. Override with env vars if needed.
DATA_DIR = os.getenv("DATA_DIR", "data")
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(DATA_DIR, "models"))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ==========================================================
# Notifications
# ==========================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==========================================================
# Account / risk
# ==========================================================
START_BALANCE = float(os.getenv("START_BALANCE", 100))

# Fraction of balance risked per trade (paper sizing). Keep this and
# RISK_PERCENT (trade_manager) consistent — see README risk section.
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.05))

# Hard daily loss stop, in account currency. Trading pauses for the day once
# daily_pnl drops below -DAILY_LOSS_LIMIT.
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", 15))

MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 10))
MAX_SIGNALS = int(os.getenv("MAX_SIGNALS", 3))

# ==========================================================
# Portfolio-level risk caps (see risk_manager.py)
# ==========================================================
# Total open risk ("heat") allowed across all positions, as % of balance.
MAX_PORTFOLIO_RISK_PCT = float(os.getenv("MAX_PORTFOLIO_RISK_PCT", 5.0))
# Max simultaneous positions in the same direction (net directional exposure).
MAX_TRADES_PER_DIRECTION = int(os.getenv("MAX_TRADES_PER_DIRECTION", 6))
# Max concurrent non-major (alt) positions — correlation-concentration proxy.
MAX_ALT_POSITIONS = int(os.getenv("MAX_ALT_POSITIONS", 8))

# Minimum model-expected R-multiple to accept a trade in AI mode. The
# expected-R regression must clear this before an otherwise-confident signal is
# taken. 0.0 = "expected to at least break even in R terms". No effect until
# the expected-R model has trained, and none at all in pure-SMC mode.
MIN_EXPECTED_R = float(os.getenv("MIN_EXPECTED_R", 0.0))

# ==========================================================
# Scan configuration
# ==========================================================
SCAN_MODE = os.getenv("SCAN_MODE", "scalp")

SCALP_BIAS_TF = "15m"
SCALP_ENTRY_TF = "5m"

SWING_BIAS_TF = "4h"
SWING_ENTRY_TF = "30m"
