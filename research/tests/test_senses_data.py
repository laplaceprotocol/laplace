import math
from pathlib import Path

import numpy as np
import pytest

from laplace_research.data.senses import FIVE_MINUTES, HOUR, hourly_realized_variance
from laplace_research.evolve.senses import carry_forward

H = 1_800_000_000 - 1_800_000_000 % HOUR  # an hour start


def test_hourly_variance_sums_twelve_five_minute_returns_from_the_previous_close() -> None:
    closes = {H - FIVE_MINUTES: 100.0} | {H + k * FIVE_MINUTES: 100.0 * math.exp(0.001 * (k + 1)) for k in range(12)}
    out = hourly_realized_variance(closes, [H])
    assert out[H] == pytest.approx(12 * 0.001**2)


def test_a_five_minute_interval_without_trades_repeats_the_previous_close() -> None:
    closes = {H - FIVE_MINUTES: 100.0, H + 6 * FIVE_MINUTES: 101.0}  # one trade in the hour
    assert hourly_realized_variance(closes, [H])[H] == pytest.approx(math.log(1.01) ** 2)


def test_an_hour_without_a_previous_close_is_left_out() -> None:
    assert hourly_realized_variance({H + FIVE_MINUTES: 100.0}, [H]) == {}


def test_settled_values_carry_forward_and_are_known_at_their_settlement() -> None:
    settlements = [(100, 0.1), (200, 0.2)]
    out = carry_forward(settlements, np.array([50, 100, 150, 200, 250], dtype=np.int64))
    assert np.isnan(out[0])
    assert list(out[1:]) == [0.1, 0.1, 0.2, 0.2]
    assert np.isnan(carry_forward([], np.array([1], dtype=np.int64))).all()


def test_failing_sources_leave_senses_unknown_when_problems_are_collected(monkeypatch: pytest.MonkeyPatch) -> None:
    from laplace_research.evolve import senses
    from laplace_research.evolve.panel import Panel

    hours = np.arange(H, H + 4 * HOUR, HOUR, dtype=np.int64)
    ones = np.ones((2, 4))
    panel = Panel(("BTC", "SOL"), hours, ones, ones, ones, ones, ones)

    def broken(*args: object, **kwargs: object) -> None:
        raise ConnectionError("down")

    monkeypatch.setattr(senses.sources, "realized_hourly", lambda *a, **k: {int(t): 0.5 for t in hours})
    monkeypatch.setattr(senses.sources, "dvol_hourly", broken)
    monkeypatch.setattr(senses.sources, "funding_rates", lambda *a, **k: [(int(hours[0]), 0.0001)])
    monkeypatch.setattr(senses.sources, "open_interest_hourly", lambda *a, **k: {int(t): 1e6 for t in hours[1:]})
    problems: list[str] = []
    out = senses.build(panel, Path("/nonexistent"), problems=problems)
    assert problems == ["dvol for BTC: ConnectionError: down"]  # SOL has no index to fetch
    assert np.isnan(out["dvol"]).all()
    assert (out["rv"] == 0.5).all()
    assert (out["funding"] == 0.0001).all()
    assert np.isnan(out["oi"][:, 0]).all()  # no snapshot for the first hour
    assert (out["oi"][:, 1:] == 1e6).all()
    with pytest.raises(ConnectionError):
        senses.build(panel, Path("/nonexistent"))


def test_open_interest_follows_the_cursor_and_keeps_only_old_snapshots(tmp_path: Path) -> None:
    import json
    import time
    from datetime import UTC, datetime

    import httpx

    from laplace_research.data.senses import open_interest_hourly

    month = datetime(2025, 10, 1, tzinfo=UTC)
    first = int(month.timestamp())
    stamps = [first + k * HOUR for k in range(5)]
    pages = {"": (stamps[3:], "next"), "next": (stamps[:3], "")}  # newest first, as the API pages

    def handler(request: httpx.Request) -> httpx.Response:
        rows, cursor = pages[request.url.params.get("cursor", "")]
        page = [{"openInterest": str(1000 + t // HOUR % 100), "timestamp": str(t * 1000)} for t in reversed(rows)]
        body = {"retCode": 0, "retMsg": "OK", "result": {"list": page, "nextPageCursor": cursor}}
        return httpx.Response(200, text=json.dumps(body))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        out = open_interest_hourly(
            "BTCUSDT", month, datetime.fromtimestamp(first + 5 * HOUR, UTC), client=client, cache_dir=tmp_path
        )
    assert sorted(out) == stamps
    assert out[stamps[2]] == 1000 + stamps[2] // HOUR % 100
    assert all(t + HOUR <= time.time() for t in out)
