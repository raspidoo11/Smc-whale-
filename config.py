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

# Label engineering: the classifier's positive label is "realized R >= this",
# not "pnl > 0". A +0.05R scratch teaches the model nothing worth repeating —
# training on meaningful wins keeps it from learning to predict fee-noise.
WIN_LABEL_MIN_R = float(os.getenv("WIN_LABEL_MIN_R", 0.5))

# Model vote in final confidence (AI mode). AI_MAX_WEIGHT is the ceiling the
# trust ramp climbs to; the ramp itself still applies — zero say below 30 real
# closed trades, full AI_MAX_WEIGHT at AI_WEIGHT_FULL_AT. Set AI_MAX_WEIGHT
# to e.g. 0.70 to let a proven model dominate the SMC score.
AI_MAX_WEIGHT = float(os.getenv("AI_MAX_WEIGHT", 0.40))
AI_WEIGHT_FULL_AT = int(os.getenv("AI_WEIGHT_FULL_AT", 150))

# ==========================================================
# Trailing stop — let winners run instead of capping at TP
# ==========================================================
# When price gets this far along the entry->TP path, cancel the hard take-profit
# and hand the position to a trailing stop. 0.90 = activate at 90% of the way.
# Defaults set from optimize.py runs (2026-07-05, BTC/ETH/SOL, 10d of 5m data,
# maker/taker fees + slippage): trail 0.3 / activation 0.90 ranked best across
# both entry modes and both fee models tested. Re-run optimize.py periodically —
# these are regime-dependent.
TRAIL_ACTIVATION_RATIO = float(os.getenv("TRAIL_ACTIVATION_RATIO", 0.90))
# Trailing distance as a percent of price (0.3 = trail 0.3% behind the peak).
TRAIL_PERCENT = float(os.getenv("TRAIL_PERCENT", 0.3))

# ==========================================================
# Entry execution — retrace limit entries vs chase-at-market
# ==========================================================
# "limit"  : place a limit order at the retracement level (FVG midpoint when a
#            fair value gap exists, else an ATR-fraction pullback) and wait for
#            price to come back to us. Better fills, but some runners leave
#            without filling.
# "market" : legacy behavior — enter at the signal candle's close immediately.
ENTRY_MODE = os.getenv("ENTRY_MODE", "limit").lower()
# Fallback retrace depth when no FVG exists: limit = entry -/+ RETRACE_ATR_FRACTION * ATR.
RETRACE_ATR_FRACTION = float(os.getenv("RETRACE_ATR_FRACTION", 0.35))
# Cancel an unfilled limit order after this many minutes (6 x 5m candles).
LIMIT_TTL_MINUTES = float(os.getenv("LIMIT_TTL_MINUTES", 30))

# ==========================================================
# Entry quality gates
# ==========================================================
# Skip entries when the bid-ask spread eats too much of the planned risk:
# reject if spread > SPREAD_MAX_FRACTION_OF_RISK * |entry - sl|.
SPREAD_MAX_FRACTION_OF_RISK = float(os.getenv("SPREAD_MAX_FRACTION_OF_RISK", 0.15))
# Pure-SMC-mode confidence bar (AI mode uses the dynamic threshold).
CONFIDENCE_REQUIRED_SMC = int(os.getenv("CONFIDENCE_REQUIRED_SMC", 40))
# Pause entries around scheduled high-impact macro events (news_filter.py).
# Off by default: the built-in calendar is approximate — enable once you've
# reviewed/edited the event windows there.
NEWS_FILTER_ENABLED = os.getenv("NEWS_FILTER_ENABLED", "false").lower() == "true"

# Sessions (UTC) in which NO NEW entries are taken — open positions keep being
# managed normally. Comma-separated. Valid names and their UTC windows:
#   asian  = 22:00-06:59 · london = 07:00-11:59 · ny = 12:00-16:59 ·
#   quiet  = 17:00-21:59
# Example: BLOCKED_SESSIONS=asian,quiet
_VALID_SESSIONS = {"asian", "london", "ny", "quiet"}
BLOCKED_SESSIONS = {
    s.strip().lower()
    for s in os.getenv("BLOCKED_SESSIONS", "").split(",")
    if s.strip()
}
_bad_sessions = BLOCKED_SESSIONS - _VALID_SESSIONS
if _bad_sessions:
    # Fail LOUDLY: a silently-ignored env var is how "I turned it off but it
    # kept trading" happens.
    raise ValueError(
        f"BLOCKED_SESSIONS contains unknown session(s) {sorted(_bad_sessions)}; "
        f"valid: {sorted(_VALID_SESSIONS)}"
    )

# ==========================================================
# Backtest realism
# ==========================================================
# Adverse slippage applied to every simulated fill, as a percent of price
# (0.02 = 2 basis points each side).
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", 0.02))
# Bybit linear-perp fee schedule. Limit fills that rest on the book pay maker;
# market orders (and SL/trailing exits, which fire at market) pay taker. When
# the gross edge per scalp is thin, this maker/taker split — not the signal —
# often decides whether a mode is profitable.
MAKER_FEE_RATE = float(os.getenv("MAKER_FEE_RATE", 0.0002))
TAKER_FEE_RATE = float(os.getenv("TAKER_FEE_RATE", 0.00055))

# ==========================================================
# Scan configuration
# ==========================================================
SCAN_MODE = os.getenv("SCAN_MODE", "scalp")

SCALP_BIAS_TF = "15m"
SCALP_ENTRY_TF = "5m"

SWING_BIAS_TF = "4h"
SWING_ENTRY_TF = "30m"
