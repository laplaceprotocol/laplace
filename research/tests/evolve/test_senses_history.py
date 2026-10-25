from pathlib import Path

import numpy as np

from laplace_research.evolve.history import HistoryReader, ensure_history, history_paths
from laplace_research.evolve.panel import Panel


def with_clock(panel: Panel) -> Panel:
    """A sense whose value at hour t is t, for every asset, and one that is NaN everywhere."""
    hours = np.broadcast_to(np.arange(len(panel.ts), dtype=np.float64), panel.close.shape).copy()
    return panel.with_senses({"clock": hours, "missing": np.full(panel.close.shape, np.nan)})


def test_senses_survive_a_round_trip_next_to_the_panel(panel: Panel, tmp_path: Path) -> None:
    path = tmp_path / "panel.npz"
    with_clock(panel).save(path)
    assert (tmp_path / "panel.senses.npz").exists()
    loaded = Panel.load(path)
    assert sorted(loaded.senses) == ["clock", "missing"]
    assert np.array_equal(loaded.senses["clock"], with_clock(panel).senses["clock"])


def test_programs_see_senses_up_to_the_forecast_hour_and_nothing_after(panel: Panel, tmp_path: Path) -> None:
    path = tmp_path / "panel.npz"
    with_clock(panel).save(path)
    reader = HistoryReader(*ensure_history(path))
    t = 900
    reader.advance_to(t)
    ctx = reader.context(0, t, 4)
    assert ctx.senses["clock"][-1] == t
    assert ctx.peer_senses["ETH"]["clock"][-1] == t
    assert np.isnan(ctx.senses["missing"]).all()
    assert len(ctx.senses["clock"]) == len(ctx.close)
    # the buffer behind every view holds nothing after hour t
    assert not reader._buffer[t + 1 :].any()


def test_history_is_rewritten_when_senses_are_added(panel: Panel, tmp_path: Path) -> None:
    path = tmp_path / "panel.npz"
    panel.save(path)
    ensure_history(path)
    assert HistoryReader(*history_paths(path)).senses == ()
    with_clock(panel).save(path)
    assert HistoryReader(*ensure_history(path)).senses == ("clock", "missing")
