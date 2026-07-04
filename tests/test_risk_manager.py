import risk_manager
from risk_manager import base_asset, trade_risk_usd, can_open_trade


def test_base_asset_parsing():
    assert base_asset("BTC/USDT:USDT") == "BTC"
    assert base_asset("ETHUSDT") == "ETH"
    assert base_asset("SOL/USDT") == "SOL"


def test_trade_risk_usd():
    t = {"entry": 100.0, "sl": 98.0, "qty": 2.0}
    assert trade_risk_usd(t) == 4.0  # |100-98| * 2


def test_rejects_duplicate_symbol():
    open_trades = [{"symbol": "BTC/USDT:USDT", "status": "OPEN", "direction": "LONG",
                    "entry": 100, "sl": 99, "qty": 1}]
    new = {"symbol": "BTC/USDT:USDT", "direction": "LONG", "entry": 100, "sl": 99, "qty": 1}
    ok, reason = can_open_trade(new, open_trades, balance=1000)
    assert not ok and "already have an open position" in reason


def test_rejects_direction_concentration(monkeypatch):
    monkeypatch.setattr(risk_manager, "MAX_TRADES_PER_DIRECTION", 2)
    open_trades = [
        {"symbol": "AAA/USDT:USDT", "status": "OPEN", "direction": "LONG", "entry": 10, "sl": 9, "qty": 1},
        {"symbol": "BBB/USDT:USDT", "status": "OPEN", "direction": "LONG", "entry": 10, "sl": 9, "qty": 1},
    ]
    new = {"symbol": "CCC/USDT:USDT", "direction": "LONG", "entry": 10, "sl": 9, "qty": 1}
    ok, reason = can_open_trade(new, open_trades, balance=100000)
    assert not ok and "LONG positions" in reason


def test_rejects_portfolio_heat(monkeypatch):
    monkeypatch.setattr(risk_manager, "MAX_PORTFOLIO_RISK_PCT", 5.0)
    monkeypatch.setattr(risk_manager, "MAX_TRADES_PER_DIRECTION", 99)
    monkeypatch.setattr(risk_manager, "MAX_ALT_POSITIONS", 99)
    # existing open risk = 4 USDT; new risk = 2 USDT; cap = 5% of 100 = 5 USDT
    open_trades = [{"symbol": "AAA/USDT:USDT", "status": "OPEN", "direction": "LONG",
                    "entry": 100, "sl": 96, "qty": 1}]  # risk 4
    new = {"symbol": "BBB/USDT:USDT", "direction": "SHORT", "entry": 100, "sl": 98, "qty": 1}  # risk 2
    ok, reason = can_open_trade(new, open_trades, balance=100)
    assert not ok and "portfolio heat" in reason


def test_allows_within_limits():
    new = {"symbol": "BTC/USDT:USDT", "direction": "LONG", "entry": 100, "sl": 99, "qty": 1}
    ok, reason = can_open_trade(new, [], balance=100000)
    assert ok and reason == "ok"
