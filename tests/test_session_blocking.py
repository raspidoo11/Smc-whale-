"""Session-blocking gate: UTC bucket mapping and the entry gate."""

from datetime import datetime, timezone

import risk_manager
from risk_manager import session_for_hour, blocked_session_now


def test_session_mapping_covers_all_24_hours():
    expected = {
        **{h: "asian" for h in (22, 23, 0, 1, 2, 3, 4, 5, 6)},
        **{h: "london" for h in range(7, 12)},
        **{h: "ny" for h in range(12, 17)},
        **{h: "quiet" for h in range(17, 22)},
    }
    for hour in range(24):
        assert session_for_hour(hour) == expected[hour], f"hour {hour}"


def test_blocked_session_detected(monkeypatch):
    monkeypatch.setattr(risk_manager, "BLOCKED_SESSIONS", {"asian"})
    # 1am WAT == 00:00 UTC -> asian: exactly the user's losing window.
    midnight_utc = datetime(2026, 7, 5, 0, 0, tzinfo=timezone.utc)
    assert blocked_session_now(midnight_utc) == "asian"
    # NY afternoon is not blocked.
    ny_time = datetime(2026, 7, 5, 14, 0, tzinfo=timezone.utc)
    assert blocked_session_now(ny_time) is None


def test_no_blocking_when_unset(monkeypatch):
    monkeypatch.setattr(risk_manager, "BLOCKED_SESSIONS", set())
    for hour in range(24):
        t = datetime(2026, 7, 5, hour, 0, tzinfo=timezone.utc)
        assert blocked_session_now(t) is None


def test_multiple_blocked_sessions(monkeypatch):
    monkeypatch.setattr(risk_manager, "BLOCKED_SESSIONS", {"london", "quiet"})
    assert blocked_session_now(datetime(2026, 7, 5, 9, 0, tzinfo=timezone.utc)) == "london"
    assert blocked_session_now(datetime(2026, 7, 5, 19, 0, tzinfo=timezone.utc)) == "quiet"
    assert blocked_session_now(datetime(2026, 7, 5, 14, 0, tzinfo=timezone.utc)) is None
    assert blocked_session_now(datetime(2026, 7, 5, 2, 0, tzinfo=timezone.utc)) is None
