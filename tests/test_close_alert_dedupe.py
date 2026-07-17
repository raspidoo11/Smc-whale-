"""Close alerts must fire once — not again from reconcile/monitor races."""

import trade_manager as tm


def test_claim_close_alert_only_once():
    tm._CLOSE_ALERT_SENT.clear()
    trade = {"trade_no": 99, "symbol": "BTC/USDT:USDT"}
    assert tm.claim_close_alert(trade) is True
    assert tm.claim_close_alert(trade) is False
    assert tm.claim_close_alert(trade_no=99) is False
    # Different trade still allowed.
    assert tm.claim_close_alert({"trade_no": 100, "symbol": "ETH/USDT:USDT"}) is True


def test_claim_close_alert_falls_back_to_symbol():
    tm._CLOSE_ALERT_SENT.clear()
    assert tm.claim_close_alert(symbol="SOL/USDT:USDT") is True
    assert tm.claim_close_alert(symbol="SOL/USDT:USDT") is False
