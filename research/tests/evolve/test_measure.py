import asyncio
import json
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from laplace_research.evolve import baselines as b
from laplace_research.evolve.db import code_hash
from laplace_research.evolve.evaluate import Evaluator, Outcome
from laplace_research.evolve.islands import SEED_PROGRAM
from laplace_research.evolve.measure import PRIMARY, collect, evaluate_all, measure, render, submit_chunks
from laplace_research.evolve.panel import Panel
from laplace_research.evolve.run import Run, RunConfig

pytest.importorskip("lightgbm")

WIDER = SEED_PROGRAM.replace("sd_hour * math.sqrt(ctx.horizon)", "1.1 * sd_hour * math.sqrt(ctx.horizon)")
BROKEN = SEED_PROGRAM.replace("return ctx.lib", "raise ValueError('no')\n    return ctx.lib")
# The seed's forecasts on weekdays; fails at weekends (all of the test holdout, none of its validation).
WEEKDAYS_ONLY = SEED_PROGRAM.replace(
    "return ctx.lib", "if ctx.weekday >= 5:\n        raise ValueError('weekend')\n    return ctx.lib"
)
# Keeps state at module level: its forecasts change with every call it has served in this instance.
STATEFUL = SEED_PROGRAM.replace(
    "def forecast(ctx):", "CALLS = []\n\n\ndef forecast(ctx):\n    CALLS.append(1)"
).replace("sd_hour * math.sqrt(ctx.horizon)", "sd_hour * math.sqrt(ctx.horizon) * (1 + len(CALLS) / 100)")


def _sensed(panel: Panel) -> Panel:
    rng = np.random.default_rng(5)
    r2 = b.hourly_returns(panel.close) ** 2
    dvol = np.full(panel.close.shape, np.nan)
    dvol[0] = 100 * np.sqrt(b.trailing_mean(r2[0], 168) * b.HOURS_PER_YEAR)
    return panel.with_senses(
        {"rv": r2 * rng.uniform(0.5, 1.5, r2.shape), "dvol": dvol, "funding": np.full(r2.shape, 1e-4)}
    )


def _run(panel: Panel, tmp_path: Path, islands: dict[str, list[tuple[str, float]]]) -> tuple[Path, dict[str, int]]:
    """A run on the synthetic panel (validation Mon-Fri, holdout from a Saturday) whose islands hold the seed and the
    given (code, search gain) programs; returns its directory and each program's id by island and position."""
    run_dir = tmp_path / "evolve-t"
    run_dir.mkdir()
    _sensed(panel).save(run_dir / "panel.npz")
    config = RunConfig(
        data_start="2025-01-01T00:00",
        data_end="2025-02-20T00:00",
        search=("2025-01-31T00:00", "2025-02-10T00:00", 7),
        validation=("2025-02-10T00:00", "2025-02-15T00:00", 5),
        holdout=("2025-02-15T00:00", "2025-02-20T00:00", 5),
    )
    (run_dir / "config.json").write_text(json.dumps(asdict(config)))
    run = Run(run_dir)
    ids = {}
    for name, programs in islands.items():
        island = run.db.add_island(name, "A hypothesis.")
        seed = run.db.add_program(
            island_id=island, parent_id=None, generation=0, mutator="seed", code=SEED_PROGRAM, gain=0.0
        )
        ids[f"{name}/seed"] = seed
        for k, (code, gain) in enumerate(programs):
            ids[f"{name}/{k}"] = run.db.add_program(
                island_id=island, parent_id=seed, generation=1, mutator="m", code=code, gain=gain
            )
    return run_dir, ids


def test_chunks_join_into_the_forecasts_of_one_evaluation(history: Path) -> None:
    hours = np.arange(800, 900, dtype=np.int64)
    with Evaluator(history, workers=2) as evaluator:
        for program in (SEED_PROGRAM, STATEFUL):
            whole = collect(submit_chunks(evaluator, program, hours, 500))
            parts = collect(submit_chunks(evaluator, program, hours, 30))
            assert isinstance(whole, np.ndarray)
            assert isinstance(parts, np.ndarray)
            np.testing.assert_array_equal(whole, parts)
        assert collect(submit_chunks(evaluator, BROKEN, hours, 30)) == "ValueError: no"


class _Flaky:
    """An evaluator whose pool breaks under the first ``crashes`` chunks submitted for each program."""

    def __init__(self, crashes: int) -> None:
        self.crashes = crashes
        self.calls: dict[str, int] = {}

    def submit(self, code: str, times: np.ndarray, time_limit_s: int) -> Future[Outcome]:
        self.calls[code] = self.calls.get(code, 0) + 1
        future: Future[Outcome] = Future()
        if self.calls[code] <= self.crashes:
            future.set_exception(BrokenProcessPool("a worker died"))
        else:
            future.set_result(Outcome(np.ones((7, 3, len(times), 10)), None, 0.0))
        return future


def test_programs_queued_on_a_crashed_pool_run_again_and_endless_crashes_stop_the_measurement() -> None:
    hours = np.arange(10, dtype=np.int64)
    out = evaluate_all(_Flaky(crashes=1), ["a", "b"], hours, 4)  # type: ignore[arg-type]
    assert set(out) == {"a", "b"}
    assert all(isinstance(v, np.ndarray) and v.shape == (7, 3, 10, 10) for v in out.values())
    with pytest.raises(RuntimeError, match="crashed 3 times"):
        evaluate_all(_Flaky(crashes=100), ["a"], hours, 4)  # type: ignore[arg-type]


def test_champions_are_chosen_on_validation_and_measured_at_every_holdout_hour(panel: Panel, tmp_path: Path) -> None:
    # the broken program is recorded as valid on the search window, yet fails once it runs on validation
    run_dir, ids = _run(panel, tmp_path, {"first": [(WIDER, 0.01)], "second": [(BROKEN, 0.5)]})

    result = measure(run_dir, workers=2, chunk_hours=40)

    assert result["hours"] == {"validation": 5 * 24 - 24, "holdout": 5 * 24 - 24}
    assert result["failures"] == {f"second#{ids['second/0']} on validation": "ValueError: no"}
    names = [c["name"] for c in result["champions"]]
    assert names[1] == f"second#{ids['second/seed']}"  # the broken program's island falls back to its next best
    assert result["candidate"] in names
    assert result["seed"]["sha256"] == code_hash(SEED_PROGRAM)
    assert (result["programs"], result["last_generation"]) == (4, 1)
    primary = result["primary"]
    assert primary["comparison"] == f"{PRIMARY[0]} vs {PRIMARY[1]}, holdout, 1h+4h"
    assert primary["interval95_72h"][0] <= primary["mean"] <= primary["interval95_72h"][1]
    assert primary["beats"] == (primary["interval95_72h"][0] > 0)
    comparisons = result["comparisons"]
    assert {"candidate vs ev-gbm", "pool(champions+ev-har) vs ev-har", "pool(champions+ev-gbm) vs ev-gbm"} <= set(
        comparisons
    )
    assert comparisons["candidate vs ev-har"]["holdout"]["hours"] == 96
    assert list(comparisons["candidate vs ev-har"]["holdout"]["months_1h4h"]) == ["2025-02"]
    for fits in result["pools"].values():
        assert set(fits) == {"h1", "h4", "h24"}
        for fit in fits.values():
            assert sum(fit["weights"].values()) == pytest.approx(1.0)
    assert {"calm", "shock", "outer bins"} <= set(result["candidate_vs_ev_har_by_segment_holdout_1h4h"])
    text = render(result)
    assert "**Primary (candidate vs ev-har, holdout, 1h+4h):**" in text
    assert "| 2025-02 (validation) | 2025-02 (holdout) |" in text

    # the holdout is scored once: a second look must say so, and the run cannot evolve on
    with pytest.raises(FileExistsError, match="measured already"):
        measure(run_dir, workers=2)
    with pytest.raises(RuntimeError, match="has been measured"):
        asyncio.run(Run(run_dir).evolve(generations=1, children=1, workers=1))
    first = (run_dir / "measure.json").read_text()
    measure(run_dir, workers=2, force=True)
    assert (run_dir / "measure-1.json").read_text() == first


def test_a_candidate_that_fails_on_the_holdout_beats_nothing_and_fails_its_pools(panel: Panel, tmp_path: Path) -> None:
    # On validation (Monday to Friday) it forecasts exactly as the seed, and as the first of equals it is chosen.
    run_dir, ids = _run(panel, tmp_path, {"first": [(WEEKDAYS_ONLY, 0.5)], "second": []})

    result = measure(run_dir, workers=2)

    assert result["candidate"] == f"first#{ids['first/0']}"
    assert result["primary"]["beats"] is False
    assert result["primary"]["failure"] == "ValueError: weekend"
    assert all("failed" in fits for fits in result["pools"].values())
    assert not any(name.startswith(("candidate", "pool")) for name in result["comparisons"])
    assert "the candidate failed on the holdout (ValueError: weekend)" in render(result)
