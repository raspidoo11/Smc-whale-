"""Trailing-stop price rounding (Bybit rejects off-tick prices)."""

from bybit_executor import round_price


def test_round_price_tick_step():
    m = {"precision": {"price": 0.0001}}
    assert round_price(0.0365567, m) == 0.0366


def test_round_price_decimal_places():
    m = {"precision": {"price": 2}}
    assert round_price(123.4567, m) == 123.46


def test_round_price_no_market_falls_back():
    assert round_price(1.23456789, None) == round(1.23456789, 6)
