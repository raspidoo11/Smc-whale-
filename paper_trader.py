import json
from config import RISK_PER_TRADE

BALANCE_FILE = "data/paper_balance.json"


def get_balance():
    try:
        with open(BALANCE_FILE, "r") as f:
            data = json.load(f)
        return float(data["balance"])
    except:
        return 100.0


def risk_amount():
    balance = get_balance()
    return balance * RISK_PER_TRADE


def calculate_qty(entry, sl):
    risk = risk_amount()
    distance = abs(entry - sl)

    if distance <= 0:
        return 0

    qty = risk / distance
    return round(qty, 6)
