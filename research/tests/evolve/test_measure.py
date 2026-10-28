import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from laplace_research.evolve import baselines as b
from laplace_research.evolve.evaluate import Evaluator
from laplace_research.evolve.islands import SEED_PROGRAM
from laplace_research.evolve.measure import PRIMARY, collect, measure, render, submit_chunks
from laplace_research.evolve.panel import Panel
from laplace_research.evolve.run import Run, RunConfig

pytest.importorskip("lightgbm")

WIDER = SEED_PROGRAM.replace("sd_hour * math.sqrt(ctx.horizon)", "1.1 * sd_hour * math.sqrt(ctx.horizon)")
BROKEN = SEED_PROGRAM.replace("return ctx.lib", "raise ValueError('no')\n    return ctx.lib")


def _sensed(panel: Panel) -> Panel:
    rng = np.random.default_rng(5)
    r2 = b.hourly_returns(panel.close) ** 2
    dvol = np.full(panel.close.shape, np.nan)
    dvol[0] = 100 * np.sqrt(b.trailing_mean(r2[0], 168) * b.HOURS_PER_YEAR)
    return panel.with_senses(
        {"rv": r2 * rng.uniform(0.5, 1.5, r2.shape), "dvol": dvol, "funding": np.full(r2.shape, 1e-4)}
    )


def test_chunks_join_into_the_forecasts_of_one_evaluation(history: Path) -> None:
    hours = np.arange(800, 900, dtype=np.int64)
    with Evaluator(history, workers=2) as evaluator:
        whole = collect(submit_chunks(evaluator, SEED_PROGRAM, hours, 500))
        parts = collect(submit_chunks(evaluator, SEED_PROGRAM, hours, 30))
        failed = collect(submit_chunks(evaluator, BROKEN, hours, 30))
    assert isinstance(whole, np.ndarray)
    assert isinstance(parts, np.ndarray)
    np.testing.assert_array_equal(whole, parts)
    assert failed == "ValueError: no"


def test_champions_are_chosen_on_validation_and_measured_at_every_holdout_hour(panel: Panel, tmp_path: Path) -> None:
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
    seeds = {}
    for name in ("first", "second"):
        island = run.db.add_island(name, "A hypothesis.")
        seeds[name] = (
            island,
            run.db.add_program(
                island_id=island, parent_id=None, generation=0, mutator="seed", code=SEED_PROGRAM, gain=0.0
            ),
        )
    (first, first_seed), (second, second_seed) = seeds["first"], seeds["second"]
    run.db.add_program(island_id=first, parent_id=first_seed, generation=1, mutator="m", code=WIDER, gain=0.01)
    # recorded as valid on the search window, yet it fails once it runs on validation
    broken = run.db.add_program(
        island_id=second, parent_id=second_seed, generation=1, mutator="m", code=BROKEN, gain=0.5
    )

    result = measure(run_dir, workers=2, chunk_hours=40)

    assert result["hours"] == {"validation": 5 * 24 - 24, "holdout": 5 * 24 - 24}
    assert result["failures"] == {f"second#{broken} on validation": "ValueError: no"}
    names = [c["name"] for c in result["champions"]]
    assert names[1] == f"second#{second_seed}"  # the broken program's island falls back to its next best
    assert result["candidate"] in names
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
    assert (run_dir / "measure.json").exists()
    assert "**Primary (candidate vs ev-har, holdout, 1h+4h):**" in render(result)
