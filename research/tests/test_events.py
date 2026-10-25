import tomllib
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest

from laplace_research.data import events

HOUR = 3600


def utc(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp())


@pytest.fixture(scope="module")
def calendar() -> events.Calendar:
    return events.load()


def times_of(calendar: events.Calendar, kind: str) -> set[int]:
    return {int(t) for t, k in zip(calendar.times, calendar.kinds, strict=True) if k == kind}


def spec() -> dict[str, Any]:
    out: dict[str, Any] = tomllib.loads(events.DEFAULT.read_text())
    return out


def test_the_calendar_holds_every_listed_release_in_time_order(calendar: events.Calendar) -> None:
    counts = {kind: calendar.kinds.count(kind) for kind in events.KINDS}
    assert (counts["fomc"], counts["cpi"], counts["nfp"]) == (24, 35, 35)
    assert counts["us-open"] == counts["us-close"] > 700
    assert np.all(np.diff(calendar.times) >= 0)


def test_eastern_times_follow_daylight_saving(calendar: events.Calendar) -> None:
    assert utc("2025-01-15T13:30") in times_of(calendar, "cpi")  # 8:30 a.m. EST
    assert utc("2025-06-11T12:30") in times_of(calendar, "cpi")  # 8:30 a.m. EDT
    assert utc("2026-10-28T18:00") in times_of(calendar, "fomc")  # 2:00 p.m., still EDT
    assert utc("2026-12-09T19:00") in times_of(calendar, "fomc")  # 2:00 p.m. EST
    opens = times_of(calendar, "us-open")
    assert utc("2026-10-30T13:30") in opens  # the Friday before daylight saving ends on 2026-11-01
    assert utc("2026-11-02T14:30") in opens  # the Monday after


def test_the_session_skips_weekends_and_holidays_and_closes_early(calendar: events.Calendar) -> None:
    opens, closes = times_of(calendar, "us-open"), times_of(calendar, "us-close")
    assert utc("2025-01-09T14:30") not in opens  # closed for the national day of mourning
    assert utc("2026-11-26T14:30") not in opens  # Thanksgiving
    assert utc("2026-10-31T13:30") not in opens  # a Saturday
    assert utc("2026-11-27T18:00") in closes  # the day after Thanksgiving closes at 1:00 p.m.
    assert utc("2026-11-27T21:00") not in closes
    assert utc("2026-11-25T21:00") in closes  # an ordinary day closes at 4:00 p.m.


def test_around_lists_events_in_hours_from_the_forecast_time(calendar: events.Calendar) -> None:
    at = utc("2026-10-28T12:00")
    around = calendar.around(at)
    assert (6.0, "fomc") in around
    assert (1.5, "us-open") in around
    assert (-16.0, "us-close") in around  # 4:00 p.m. the day before
    assert [h for h, _ in around] == sorted(h for h, _ in around)
    assert all(-events.BEFORE_HOURS <= h < events.AFTER_HOURS for h, _ in around)
    assert (-24.0, "fomc") in calendar.around(at + 30 * HOUR)  # the start of the window is included
    assert all(k != "fomc" for _, k in calendar.around(at - 42 * HOUR))  # 48 hours ahead is not


def test_around_is_empty_where_the_calendar_does_not_reach(calendar: events.Calendar) -> None:
    assert calendar.around(utc("2024-12-31T12:00")) == ()
    assert calendar.around(utc("2027-12-30T12:00")) == ()
    assert calendar.around(utc("2027-12-28T12:00")) != ()


def test_count_counts_events_of_some_kinds_in_half_open_windows(calendar: events.Calendar) -> None:
    lo = np.array([utc("2026-10-28T12:00"), utc("2026-10-28T18:00"), utc("2026-10-28T19:00")])
    assert calendar.count(events.RELEASES, lo, lo + HOUR).tolist() == [0, 1, 0]
    assert calendar.count(events.RELEASES, lo, lo + 7 * HOUR).tolist() == [1, 1, 0]
    assert calendar.count(("us-open",), lo, lo + 24 * HOUR).tolist() == [1, 1, 1]


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("cpi", "dates", ["2025-01-15", "2028-01-12"], "outside the covered range"),
        ("nfp", "dates", ["2025-01-11"], "weekend"),
        ("fomc", "dates", ["2025-03-19", "2025-01-29"], "ascending"),
    ],
)
def test_parse_refuses_impossible_dates(section: str, key: str, value: list[str], message: str) -> None:
    broken = spec()
    broken[section][key] = value
    with pytest.raises(ValueError, match=message):
        events.parse(broken)


def test_parse_refuses_a_day_that_is_both_a_holiday_and_an_early_close() -> None:
    broken = spec()
    broken["nyse"]["early_closes"] = ["2025-01-20"]
    with pytest.raises(ValueError, match="both"):
        events.parse(broken)


def test_session_seconds_accumulate_only_while_the_session_is_open(calendar: events.Calendar) -> None:
    at = np.array([utc(x) for x in ("2026-10-28T13:00", "2026-10-28T13:30", "2026-10-28T16:00", "2026-10-28T23:00")])
    seconds = calendar.session_seconds(at)
    assert np.diff(seconds).tolist() == [0.0, 2.5 * HOUR, 4.0 * HOUR]  # 13:30 to 20:00 UTC is the session


def test_overlay_inputs_by_hand(calendar: events.Calendar) -> None:
    from laplace_research.evolve.baselines import overlay_features, overlay_names

    def inputs(at: str, h: int) -> dict[str, float]:
        x = overlay_features(calendar, np.array([utc(at)]), h)[0]
        return dict(zip(overlay_names(h), x.tolist(), strict=True))

    noon = inputs("2026-10-28T12:00", 1)  # Wednesday, FOMC at 18:00 UTC, NYSE 13:30 to 20:00 UTC
    assert noon["hour-12"] == pytest.approx(23 / 24)
    assert noon["hour-3"] == pytest.approx(-1 / 24)
    assert [noon[k] for k in ("weekend", "releases", "recent-releases", "upcoming-releases", "us-open")] == [0] * 5
    four = inputs("2026-10-28T12:00", 4)
    assert (four["us-open"], four["us-session"], four["upcoming-releases"]) == (1.0, 0.625, 0.0)
    day = inputs("2026-10-28T12:00", 24)
    assert (day["releases"], day["us-open"], day["us-session"]) == (1.0, 1.0, pytest.approx(6.5 / 24))
    assert "hour-12" not in day
    assert inputs("2026-10-28T17:00", 1)["upcoming-releases"] == 1.0
    assert inputs("2026-10-28T17:00", 1)["us-session"] == 1.0
    after = inputs("2026-10-28T20:00", 1)
    assert (after["recent-releases"], after["us-session"]) == (1.0, 0.0)
    assert inputs("2026-10-30T12:00", 24)["weekend"] == 0.5  # Friday noon to Saturday noon
    early = inputs("2024-12-31T12:00", 4)  # before the calendar: only the clock-based inputs remain
    assert early["hour-12"] == pytest.approx(23 / 24)
    assert [early[k] for k in ("releases", "us-open", "us-session")] == [0, 0, 0]


def test_the_empty_calendar_lists_nothing_and_a_loaded_one_knows_its_file(calendar: events.Calendar) -> None:
    at = utc("2026-10-28T12:00")
    assert events.EMPTY.around(at) == ()
    assert events.EMPTY.count(events.RELEASES, np.array([at]), np.array([at + 24 * HOUR])).tolist() == [0]
    assert calendar.source == events.DEFAULT
    assert calendar.raw == events.DEFAULT.read_bytes()
