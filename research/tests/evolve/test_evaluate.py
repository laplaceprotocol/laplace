from pathlib import Path

import numpy as np
import pytest

from laplace_research.evolve.baselines import naive
from laplace_research.evolve.context import LOOKBACK
from laplace_research.evolve.evaluate import (
    MAX_HORIZON,
    Evaluator,
    log_scores,
    mean_gain,
    ranked_probability_scores,
    realized_bins,
    sample_times,
)
from laplace_research.evolve.feed import N_BINS, finalize
from laplace_research.evolve.history import HistoryReader, history_paths, write_history
from laplace_research.evolve.islands import SEED_PROGRAM
from laplace_research.evolve.panel import Panel, utc

# Uses every peer and every field, so any future data that leaked into a context would change its output.
PEEKING_PROGRAM = """\
import numpy as np


def forecast(ctx):
    total = 0.0
    for name in ctx.peers:
        c = ctx.peers[name]
        total += float(np.sum(c.close) + np.sum(c.high) + np.sum(c.low) + np.sum(c.open) + np.sum(c.volume))
    out = np.ones(10)
    out[int(abs(total) * 1000) % 10] += 1.0
    return out
"""


def test_perfect_forecast_has_zero_rps_and_near_zero_log_loss() -> None:
    realized = np.array([[[3]]])
    probs = np.zeros((1, 1, 1, N_BINS))
    probs[..., 3] = 1
    final = finalize(probs)
    assert ranked_probability_scores(final, realized)[0, 0, 0] == pytest.approx(0, abs=1e-6)
    assert log_scores(final, realized)[0, 0, 0] == pytest.approx(0, abs=2e-3)


def test_scores_never_smooth_final_forecasts_again() -> None:
    final = finalize(np.eye(N_BINS)[None, None, 3:4])
    pooled = 0.5 * final + 0.5 * final
    realized = np.array([[[7]]])
    assert log_scores(pooled, realized)[0, 0, 0] == pytest.approx(log_scores(final, realized)[0, 0, 0])


def test_window_labels_settle_before_the_window_ends(panel: Panel) -> None:
    end = utc(int(panel.ts[1000]))
    times = sample_times(panel, utc(int(panel.ts[0])), end, 1)
    assert times[0] >= LOOKBACK - 1
    assert times[-1] + MAX_HORIZON < panel.index_at_or_after(end)
    assert times[-1] + MAX_HORIZON == panel.index_at_or_after(end) - 1


def test_reader_buffer_holds_nothing_after_the_forecast_hour(history: Path) -> None:
    reader = HistoryReader(*history_paths(history))
    reader.advance_to(800)
    ctx = reader.context(0, 800, 24)
    assert ctx.close.base is not None
    assert reader.loaded == 801
    assert np.all(ctx.close.base[801:] == 0)
    with pytest.raises(ValueError, match="advance_to"):
        reader.context(0, 801, 24)


def test_forecasts_do_not_depend_on_the_future(panel: Panel, tmp_path: Path) -> None:
    """Rewrite every candle after a cutoff; forecasts made before it must not change."""
    cutoff = 1000
    rng = np.random.default_rng(1)
    noisy = {f: getattr(panel, f).copy() for f in ("open", "high", "low", "close", "volume")}
    for arr in noisy.values():
        arr[:, cutoff:] *= rng.uniform(0.5, 1.5, arr[:, cutoff:].shape)
    altered = Panel(assets=panel.assets, ts=panel.ts.copy(), **noisy)
    paths = []
    for name, p in (("original", panel), ("altered", altered)):
        path = tmp_path / name / "panel.npz"
        path.parent.mkdir()
        p.save(path)
        write_history(p, path)
        paths.append(path)
    times = np.arange(LOOKBACK - 1, cutoff, 37, dtype=np.int64)
    outputs = []
    for path in paths:
        with Evaluator(path, workers=1) as evaluator:
            outcome = evaluator.run(PEEKING_PROGRAM, times, 60)
        assert outcome.error is None
        outputs.append(outcome.probs)
    assert np.array_equal(outputs[0], outputs[1])


def test_evaluator_scores_the_seed_at_zero_gain_and_reports_failures(panel: Panel, history: Path) -> None:
    times = sample_times(panel, utc(int(panel.ts[0])), utc(int(panel.ts[-1])), 50)
    realized = realized_bins(panel, times)
    reference = log_scores(finalize(naive(panel.close).probs(times)), realized)
    with Evaluator(history, workers=2) as evaluator:
        seed = evaluator.run(SEED_PROGRAM, times, 30)
        assert seed.error is None
        assert seed.probs is not None
        assert mean_gain(log_scores(finalize(seed.probs), realized), reference) == pytest.approx(0, abs=1e-9)

        crash = evaluator.run("def forecast(ctx):\n    return [1] * 3\n", times, 30)
        assert crash.probs is None
        assert crash.error is not None
        assert "bad output" in crash.error

        slow = evaluator.run("def forecast(ctx):\n    while True:\n        pass\n", times, 1)
        assert slow.error is not None
        assert "timed out" in slow.error

        write = evaluator.run("def forecast(ctx):\n    c = ctx.close\n    c[0] = 1.0\n    return [1] * 10\n", times, 30)
        assert write.error is not None
        assert "read-only" in write.error


def test_numpy_methods_work_in_a_fresh_worker(panel: Panel, history: Path) -> None:
    """numpy's C code imports some of its own submodules lazily, through the builtins of the running frame (the
    program's); the restricted import must let those through while the static check still refuses them in source."""
    program = (
        "def forecast(ctx):\n"
        "    c = ctx.close\n"
        "    x = c.clip(0.0, 1e12).mean() + c.std() + c.var() + c.round(2).max() + c.min() + float(c.any())\n"
        "    return [1.0 + 0.0 * x] * 10\n"
    )
    times = sample_times(panel, utc(int(panel.ts[0])), utc(int(panel.ts[-1])), 200)
    with Evaluator(history, workers=1) as evaluator:
        outcome = evaluator.run(program, times, 60)
    assert outcome.error is None


def test_workers_list_events_from_their_calendar_and_survive_a_broken_one(
    panel: Panel, history: Path, tmp_path: Path
) -> None:
    program = "def forecast(ctx):\n    return [1.0 + len(ctx.events)] + [1.0] * 9\n"
    times = np.array([900], dtype=np.int64)  # 2025-02-07 12:00 UTC, in the calendar's range
    with Evaluator(history, workers=1) as evaluator:
        listed = evaluator.run(program, times, 30)
    with Evaluator(history, workers=1, calendar=None) as evaluator:
        unlisted = evaluator.run(program, times, 30)
    broken = tmp_path / "events.toml"
    broken.write_text("covers = [")
    with Evaluator(history, workers=1, calendar=broken) as evaluator:
        failed = evaluator.run(program, times, 30)
    assert listed.probs is not None
    assert unlisted.probs is not None
    assert listed.probs[0, 0, 0, 0] > 1.0
    assert unlisted.probs[0, 0, 0, 0] == 1.0
    assert failed.probs is None
    assert failed.error is not None
    assert failed.error.startswith("calendar:")


def test_every_forecast_hour_starts_from_a_fresh_instance_of_the_program(history: Path) -> None:
    # As the live engine runs a program once per issue: state kept at module level lasts one hour, never longer.
    counting = "CALLS = []\n\n\ndef forecast(ctx):\n    CALLS.append(1)\n    return [float(len(CALLS))] + [1.0] * 9\n"
    times = np.arange(800, 805, dtype=np.int64)
    with Evaluator(history, workers=1) as evaluator:
        outcome = evaluator.run(counting, times, 30)
    assert outcome.probs is not None
    calls = outcome.probs[..., 0]  # [asset, horizon, time]: the call's number within its hour
    expected = np.arange(7)[:, None, None] * 3 + np.arange(3)[None, :, None] + 1
    np.testing.assert_array_equal(calls, np.broadcast_to(expected, calls.shape))
