from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from laplace_research.data import events
from laplace_research.evolve.history import HistoryReader, ensure_history
from laplace_research.evolve.panel import Panel


def index_of(panel: Panel, candle_open: str) -> int:
    return int(np.searchsorted(panel.ts, int(datetime.fromisoformat(candle_open).replace(tzinfo=UTC).timestamp())))


def test_contexts_list_the_events_around_their_forecast_time(panel: Panel, history: Path) -> None:
    reader = HistoryReader(*ensure_history(history))
    t = index_of(panel, "2025-02-07T11:00")  # the forecast is made at 12:00 UTC on a jobs-report Friday
    reader.advance_to(t)
    ctx = reader.context(0, t, 4)
    assert (1.5, "nfp") in ctx.events  # 8:30 a.m. EST
    assert (2.5, "us-open") in ctx.events
    assert ctx.events == reader.context(5, t, 24).events  # the same for every asset and horizon
    reader.advance_to(t + 1)
    assert (0.5, "nfp") in reader.context(0, t + 1, 1).events


def test_a_reader_can_be_given_another_calendar(panel: Panel, history: Path) -> None:
    spec = {
        "covers": ["2025-01-01", "2025-12-31"],
        "fomc": {"time": "14:00", "dates": []},
        "cpi": {"time": "08:30", "dates": []},
        "nfp": {"time": "08:30", "dates": []},
        "nyse": {
            "open": "09:30",
            "close": "16:00",
            "early_close": "13:00",
            "holidays": ["2025-02-07"],
            "early_closes": [],
        },
    }
    reader = HistoryReader(*ensure_history(history), calendar=events.parse(spec))
    t = index_of(panel, "2025-02-07T11:00")
    reader.advance_to(t)
    kinds = {k for h, k in reader.context(0, t, 1).events if 0 <= h < 24}
    assert kinds == set()  # no releases listed, and the session is closed that Friday
