"""The candles an issue runs on: for every feed asset, the hours that closed before the issue hour."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from laplace_research.data.candles import HOUR, fetch_hourly
from laplace_research.evolve.context import LOOKBACK
from laplace_research.evolve.feed import product
from laplace_research.evolve.panel import Panel, build

WINDOW_HOURS = LOOKBACK + 24  # a day of spare history beyond what programs see
SENTINEL = "BTC"  # trades every second; its first candle of the issue hour shows the previous hour has closed
GRACE = timedelta(minutes=5)  # after this, the exchange's own clock vouches for the hour if its candle feed lags
EXCHANGE_TIME = "https://api.exchange.coinbase.com/time"

log = logging.getLogger(__name__)


class CandlesNotReady(RuntimeError):
    """The exchange has not yet published, for every asset, the last candle before the issue hour."""


def issue_time(issue: int) -> datetime:
    """When an issue's forecasts are made: the start of its hour, once the previous candle has closed."""
    return datetime.fromtimestamp(issue * HOUR, UTC)


def current_issue(now: datetime | None = None) -> int:
    return int((now or datetime.now(UTC)).timestamp()) // HOUR


def hour_started(issue: int, cache_dir: Path) -> bool:
    """Whether the exchange has opened the issue hour's candle, i.e. the previous hour is closed by its own clock."""
    start = issue_time(issue)
    with httpx.Client(timeout=30.0, headers={"User-Agent": "laplace-engine"}) as client:
        rows = fetch_hourly(product(SENTINEL), start, start + timedelta(hours=1), client=client, cache_dir=cache_dir)
    return any(c.ts == issue * HOUR for c in rows)


def exchange_time() -> datetime:
    """The exchange's own clock."""
    with httpx.Client(timeout=15.0, headers={"User-Agent": "laplace-engine"}) as client:
        response = client.get(EXCHANGE_TIME)
        response.raise_for_status()
        return datetime.fromtimestamp(float(response.json()["epoch"]), UTC)


def fetch(issue: int, cache_dir: Path, *, now: datetime | None = None) -> Panel:
    """The ``WINDOW_HOURS`` hours before the issue, ending with the candle that closed at the issue time.

    For the current issue the exchange must also vouch that the issue hour has begun, so a fast local clock can never
    seal on a candle that is still trading: normally by having opened the hour's first candle; if its candle feed
    lags past ``GRACE``, by its own clock. The committed last candle is then the definitive starting price.
    """
    end = issue_time(issue)
    if issue >= current_issue(now) and not hour_started(issue, cache_dir):
        if (now or datetime.now(UTC)) < end + GRACE or exchange_time() < end + timedelta(seconds=30):
            raise CandlesNotReady("the exchange has not opened the issue hour yet")
        log.warning("issue %d: the exchange's new-hour candle is late; its clock says the hour has closed", issue)
    panel = build(end - timedelta(hours=WINDOW_HOURS), end, cache_dir)
    if len(panel.ts) < LOOKBACK:
        raise CandlesNotReady(f"only {len(panel.ts)} shared hours before issue {issue}")
    if int(panel.ts[-1]) != (issue - 1) * HOUR:
        last = datetime.fromtimestamp(int(panel.ts[-1]), UTC)
        raise CandlesNotReady(f"the last candle every asset has opened at {last:%Y-%m-%d %H:%M}")
    return panel


def wait_for(issue: int, cache_dir: Path, *, deadline: datetime, poll_seconds: float = 10.0) -> Panel:
    """Fetch until the issue's candles are complete; past ``deadline`` the last error is raised."""
    while True:
        try:
            return fetch(issue, cache_dir)
        except (CandlesNotReady, httpx.HTTPError, ValueError, RuntimeError) as exc:  # RuntimeError: rate limited
            if datetime.now(UTC) + timedelta(seconds=poll_seconds) > deadline:
                raise
            log.info("issue %d: candles not ready (%s); retrying in %.0fs", issue, exc, poll_seconds)
            time.sleep(poll_seconds)
