import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

START_BALANCE = float(os.getenv("START_BALANCE", 100))

RISK_PER_TRADE = 0.05

MAX_SIGNALS = 3

SCAN_MODE = os.getenv("SCAN_MODE", "scalp")

SCALP_BIAS_TF = "15m"
SCALP_ENTRY_TF = "5m"

SWING_BIAS_TF = "4h"
SWING_ENTRY_TF = "30m"
