"""Bybit execution guards: qty/price formatting and live risk sizing.

These catch the two failures that silently blocked live/demo orders on
Lightsail/Railway:
  1. float step math producing '0.30000000000000004' (Bybit ErrCode 10001)
  2. risk sized from START_BALANCE paper seed instead of wallet equity,
     so BTC qty rounded to 0 under min lot size
"""

from decimal import Decimal

import bybit_executor as be


def _sol_market():
    return {
        "precision": {"amount": 0.1, "price": 0.01},
        "limits": {"amount": {"min": 0.1, "max": 96000.0}},
        "info": {
            "lotSizeFilter": {
                "qtyStep": "0.1",
                "minOrderQty": "0.1",
                "minNotionalValue": "5",
            },
            "priceFilter": {"tickSize": "0.01"},
        },
    }


def _btc_market():
    return {
        "precision": {"amount": 0.001, "price": 0.1},
        "limits": {"amount": {"min": 0.001, "max": 1500.0}},
        "info": {
            "lotSizeFilter": {
                "qtyStep": "0.001",
                "minOrderQty": "0.001",
                "minNotionalValue": "5",
            },
            "priceFilter": {"tickSize": "0.1"},
        },
    }


def test_format_decimal_strips_float_noise():
    # The exact string Bybit rejected in reproduction.
    noisy = Decimal("0.1") * 3  # 0.30000000000000004 as float path
    # Via our quantize path (step 0.1):
    qty, s = be.quantize_qty(0.3, _sol_market())
    assert qty == 0.3
    assert s == "0.3"
    assert "000000" not in s
    assert "e" not in s.lower()


def test_quantize_qty_floors_to_step():
    qty, s = be.quantize_qty(1.29, _sol_market())
    assert qty == 1.2
    assert s == "1.2"


def test_quantize_qty_zero_when_below_step():
    qty, s = be.quantize_qty(0.05, _sol_market())
    assert qty is None
    assert s is None


def test_quantize_price_tick():
    price, s = be.quantize_price(64017.67, _btc_market())
    assert price == 64017.7
    assert s == "64017.7"


def test_round_price_uses_tick_filter():
    assert be.round_price(0.0365567, {
        "precision": {"price": 0.0001},
        "info": {"priceFilter": {"tickSize": "0.0001"}},
    }) == 0.0366


def test_live_risk_uses_wallet_equity(monkeypatch):
    """With EXECUTE_TRADES, base risk must come from wallet equity, not the
    $100 paper seed — otherwise BTC qty is 0 and every major is skipped."""
    monkeypatch.setattr(be, "EXECUTE_TRADES", True)
    monkeypatch.setattr(be, "get_wallet_balance_usdt", lambda: 100_000.0)
    monkeypatch.setenv("RISK_PERCENT", "0.5")
    # 0.5% of 100k = $500
    assert be._live_risk_base() == 500.0


def test_live_risk_falls_back_to_paper_when_wallet_unavailable(monkeypatch):
    monkeypatch.setattr(be, "EXECUTE_TRADES", True)
    monkeypatch.setattr(be, "get_wallet_balance_usdt", lambda: None)
    monkeypatch.setattr(be, "get_risk_amount", lambda: 12.34)
    assert be._live_risk_base() == 12.34


def test_paper_risk_ignores_wallet(monkeypatch):
    monkeypatch.setattr(be, "EXECUTE_TRADES", False)
    monkeypatch.setattr(be, "get_risk_amount", lambda: 0.5)
    # Wallet would be huge, but paper mode must not touch it.
    monkeypatch.setattr(be, "get_wallet_balance_usdt", lambda: 100_000.0)
    assert be._live_risk_base() == 0.5


def test_calculate_proper_qty_works_for_btc_with_live_equity(monkeypatch):
    monkeypatch.setattr(be, "EXECUTE_TRADES", True)
    monkeypatch.setattr(be, "get_wallet_balance_usdt", lambda: 100_000.0)
    monkeypatch.setattr(be, "get_symbol_info", lambda _s: _btc_market())
    monkeypatch.setattr(be, "get_ai_risk_percent", lambda *a, **k: 0.5)
    monkeypatch.setenv("RISK_PERCENT", "0.5")
    # $500 risk, $640 risk/unit (~1% of 64k) -> ~0.781 BTC -> 0.781 stepped
    qty = be.calculate_proper_qty("BTCUSDT", 64000.0, 63360.0, ai_prob=50)
    assert qty is not None
    assert qty >= 0.001
    _, s = be.quantize_qty(qty, _btc_market())
    assert s is not None
    assert "e" not in s.lower()
    assert "000000" not in s


def test_calculate_proper_qty_none_when_paper_seed_too_small(monkeypatch):
    """Reproduce the Lightsail bug: $100 seed + BTC wide stop => qty 0."""
    monkeypatch.setattr(be, "EXECUTE_TRADES", True)
    # Wallet unavailable, paper risk ~$0.50
    monkeypatch.setattr(be, "get_wallet_balance_usdt", lambda: None)
    monkeypatch.setattr(be, "get_risk_amount", lambda: 0.50)
    monkeypatch.setattr(be, "get_symbol_info", lambda _s: _btc_market())
    monkeypatch.setattr(be, "get_ai_risk_percent", lambda *a, **k: 0.3)  # low conf
    qty = be.calculate_proper_qty("BTCUSDT", 64000.0, 62000.0, ai_prob=40)
    assert qty is None
