"""Scheduled events, known before they happen: FOMC decisions, CPI and jobs-report releases, and the open and close of
the NYSE regular session.

The dates are copied from their publishers' calendars into ``events.toml``, which names each source. Times are the
publishers' US Eastern Time, converted with the IANA zone America/New_York, so daylight saving is exact. A calendar
covers a stated range of dates and knows nothing outside it.

Programs see the events around their forecast time as ``ctx.events`` (see ``evolve.context``): when each event is
scheduled, never what it announces.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt

DEFAULT = Path(__file__).with_name("events.toml")
EASTERN = ZoneInfo("America/New_York")
RELEASES = ("fomc", "cpi", "nfp")
KINDS = (*RELEASES, "us-open", "us-close")
HOUR = 3600
BEFORE_HOURS = 24  # how far back a context's events reach
AFTER_HOURS = 48  # and how far ahead

Events = tuple[tuple[float, str], ...]
IntArray = npt.NDArray[np.int64]


def eastern(day: date, at: time) -> int:
    """Unix seconds of a US Eastern wall-clock time."""
    return int(datetime.combine(day, at, tzinfo=EASTERN).timestamp())


@dataclass(frozen=True)
class Calendar:
    first: date  # the first and last covered days (Eastern Time), inclusive
    last: date
    times: IntArray  # unix seconds, ascending
    kinds: tuple[str, ...]
    source: Path | None = None  # the file it was read from
    raw: bytes = b""  # that file's bytes, so its hash names exactly what was read

    @property
    def start(self) -> int:
        return eastern(self.first, time(0))

    @property
    def end(self) -> int:
        return eastern(self.last + timedelta(days=1), time(0))

    def around(self, at: int) -> Events:
        """Every event from ``BEFORE_HOURS`` before ``at`` (inclusive) to ``AFTER_HOURS`` after it (exclusive), as
        (hours from ``at``, kind) in time order; empty unless the calendar covers all of that span."""
        lo, hi = at - BEFORE_HOURS * HOUR, at + AFTER_HOURS * HOUR
        if lo < self.start or hi > self.end:
            return ()
        i, j = np.searchsorted(self.times, [lo, hi], side="left")
        return tuple(((int(self.times[k]) - at) / HOUR, self.kinds[k]) for k in range(int(i), int(j)))

    def count(self, kinds: tuple[str, ...], lo: IntArray, hi: IntArray) -> IntArray:
        """For each pair of instants, the number of events of ``kinds`` with ``lo <= time < hi``."""
        stamps = self.times[np.isin(np.array(self.kinds), kinds)]
        out: IntArray = np.searchsorted(stamps, hi, side="left") - np.searchsorted(stamps, lo, side="left")
        return out


def _days(values: list[str], first: date, last: date, what: str) -> list[date]:
    days = [date.fromisoformat(v) for v in values]
    if days != sorted(set(days)):
        raise ValueError(f"{what}: dates must be ascending and distinct")
    outside = [d for d in days if not first <= d <= last]
    if outside:
        raise ValueError(f"{what}: {outside[0]} is outside the covered range")
    weekend = [d for d in days if d.weekday() >= 5]
    if weekend:
        raise ValueError(f"{what}: {weekend[0]} is a weekend")
    return days


def parse(spec: dict[str, Any], *, source: Path | None = None, raw: bytes = b"") -> Calendar:
    first, last = (date.fromisoformat(d) for d in spec["covers"])
    events: list[tuple[int, str]] = []
    for kind in RELEASES:
        at = time.fromisoformat(spec[kind]["time"])
        events += [(eastern(d, at), kind) for d in _days(spec[kind]["dates"], first, last, kind)]
    nyse = spec["nyse"]
    holidays = set(_days(nyse["holidays"], first, last, "nyse holidays"))
    early = set(_days(nyse["early_closes"], first, last, "nyse early closes"))
    if holidays & early:
        raise ValueError("a day cannot be both a holiday and an early close")
    opens, closes, early_close = (time.fromisoformat(nyse[k]) for k in ("open", "close", "early_close"))
    day = first
    while day <= last:
        if day.weekday() < 5 and day not in holidays:
            events.append((eastern(day, opens), "us-open"))
            events.append((eastern(day, early_close if day in early else closes), "us-close"))
        day += timedelta(days=1)
    events.sort()
    times = np.array([t for t, _ in events], dtype=np.int64)
    return Calendar(first, last, times, tuple(k for _, k in events), source, raw)


@cache
def load(path: Path = DEFAULT) -> Calendar:
    raw = path.read_bytes()
    return parse(tomllib.loads(raw.decode()), source=path, raw=raw)


EMPTY = Calendar(date(1970, 1, 1), date(1970, 1, 1), np.zeros(0, dtype=np.int64), ())  # covers nothing, lists nothing
