"""Bybit's tokenized commodity/TradFi perps (CLUSDT crude oil, XAUUSDT gold,
...) fail order placement with ErrCode 110125 unless special terms are
accepted — the scanner must never propose them."""

import sys
import types

if "exchange" not in sys.modules or not hasattr(sys.modules.get("exchange"), "_TEST_STUB"):
    stub = types.ModuleType("exchange")
    stub._TEST_STUB = True
    stub.get_exchange = lambda: None
    stub.get_trade_client = lambda: None
    sys.modules["exchange"] = stub

from scanner import is_commodity_pair


def test_blocks_commodity_perps():
    assert is_commodity_pair("CL/USDT:USDT") is True      # crude oil (the 110125 error)
    assert is_commodity_pair("CLUSDT") is True
    assert is_commodity_pair("XAU/USDT:USDT") is True     # gold
    assert is_commodity_pair("NGUSDT") is True            # nat gas
    assert is_commodity_pair("SPXUSDT") is True           # index


def test_allows_real_crypto_including_lookalikes():
    # Exact-base matching: crypto whose ticker merely CONTAINS a commodity
    # code must never be caught.
    assert is_commodity_pair("BTC/USDT:USDT") is False
    assert is_commodity_pair("CLVUSDT") is False   # Clover
    assert is_commodity_pair("XAIUSDT") is False   # XAI
    assert is_commodity_pair("SOLUSDT") is False
