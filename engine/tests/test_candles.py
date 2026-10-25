from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from laplace_engine import candles
from laplace_research.evolve.panel import Panel

from .conftest import ISSUE


def test_issue_numbers_are_hours_since_the_epoch() -> None:
    assert candles.issue_time(ISSUE) == datetime(2026, 10, 23, 7, tzinfo=UTC)
    assert candles.current_issue(datetime(2026, 10, 23, 7, 59, 59, tzinfo=UTC)) == ISSUE


def test_fetch_requires_the_candle_that_closed_at_the_issue_time(
    monkeypatch: pytest.MonkeyPatch, make_panel: Callable[..., Panel], tmp_path: Path
) -> None:
    requested = []

    def build(start: datetime, end: datetime, cache_dir: Path) -> Panel:
        requested.append((start, end))
        return make_panel(end_issue=ISSUE - 1)  # the exchange has not published the last hour yet

    monkeypatch.setattr(candles, "build", build)
    with pytest.raises(candles.CandlesNotReady, match="05:00"):
        candles.fetch(ISSUE, tmp_path)
    assert requested == [(candles.issue_time(ISSUE) - timedelta(hours=candles.WINDOW_HOURS), candles.issue_time(ISSUE))]

    monkeypatch.setattr(candles, "build", lambda start, end, cache_dir: make_panel(end_issue=ISSUE))
    assert int(candles.fetch(ISSUE, tmp_path).ts[-1]) == (ISSUE - 1) * 3600


def test_wait_for_retries_until_the_deadline(
    monkeypatch: pytest.MonkeyPatch, make_panel: Callable[..., Panel], tmp_path: Path
) -> None:
    calls = []

    def build(start: datetime, end: datetime, cache_dir: Path) -> Panel:
        calls.append(start)
        return make_panel(end_issue=ISSUE if len(calls) >= 3 else ISSUE - 1)

    monkeypatch.setattr(candles, "build", build)
    deadline = datetime.now(UTC) + timedelta(seconds=5)
    assert candles.wait_for(ISSUE, tmp_path, deadline=deadline, poll_seconds=0.01).ts[-1] == (ISSUE - 1) * 3600
    assert len(calls) == 3
    calls.clear()
    monkeypatch.setattr(candles, "build", lambda start, end, cache_dir: make_panel(end_issue=ISSUE - 1))
    with pytest.raises(candles.CandlesNotReady):
        candles.wait_for(ISSUE, tmp_path, deadline=datetime.now(UTC), poll_seconds=0.01)


def test_wait_for_rides_out_rate_limits(
    monkeypatch: pytest.MonkeyPatch, make_panel: Callable[..., Panel], tmp_path: Path
) -> None:
    calls = []

    def build(start: datetime, end: datetime, cache_dir: Path) -> Panel:
        calls.append(start)
        if len(calls) < 2:
            raise RuntimeError("BTC-USD: rate-limited on every attempt")
        return make_panel(end_issue=ISSUE)

    monkeypatch.setattr(candles, "build", build)
    deadline = datetime.now(UTC) + timedelta(seconds=5)
    assert candles.wait_for(ISSUE, tmp_path, deadline=deadline, poll_seconds=0.01).ts[-1] == (ISSUE - 1) * 3600


def test_the_current_issue_waits_until_the_exchange_opens_its_hour(
    monkeypatch: pytest.MonkeyPatch, make_panel: Callable[..., Panel], tmp_path: Path
) -> None:
    monkeypatch.setattr(candles, "current_issue", lambda now=None: ISSUE)
    monkeypatch.setattr(candles, "build", lambda start, end, cache_dir: make_panel(end_issue=ISSUE))
    monkeypatch.setattr(candles, "hour_started", lambda issue, cache_dir: False)
    monkeypatch.setattr(candles, "exchange_time", lambda: pytest.fail("inside the grace period: no clock check"))
    early = candles.issue_time(ISSUE) + timedelta(seconds=40)
    with pytest.raises(candles.CandlesNotReady, match="not opened"):
        candles.fetch(ISSUE, tmp_path, now=early)
    monkeypatch.setattr(candles, "hour_started", lambda issue, cache_dir: True)
    assert int(candles.fetch(ISSUE, tmp_path, now=early).ts[-1]) == (ISSUE - 1) * 3600


def test_a_lagging_candle_feed_falls_back_to_the_exchange_clock_after_the_grace(
    monkeypatch: pytest.MonkeyPatch, make_panel: Callable[..., Panel], tmp_path: Path
) -> None:
    monkeypatch.setattr(candles, "build", lambda start, end, cache_dir: make_panel(end_issue=ISSUE))
    monkeypatch.setattr(candles, "hour_started", lambda issue, cache_dir: False)
    start = candles.issue_time(ISSUE)
    monkeypatch.setattr(candles, "exchange_time", lambda: start + timedelta(minutes=6))
    with pytest.raises(candles.CandlesNotReady):  # inside the grace period, keep waiting for the candle
        candles.fetch(ISSUE, tmp_path, now=start + timedelta(minutes=2))
    assert int(candles.fetch(ISSUE, tmp_path, now=start + timedelta(minutes=6)).ts[-1]) == (ISSUE - 1) * 3600
    monkeypatch.setattr(candles, "exchange_time", lambda: start - timedelta(seconds=5))
    with pytest.raises(candles.CandlesNotReady):  # our clock is fast: the exchange says the hour has not begun
        candles.fetch(ISSUE, tmp_path, now=start + timedelta(minutes=6))
