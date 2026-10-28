import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from laplace_research.evolve import baselines as b
from laplace_research.evolve.evaluate import Evaluator
from laplace_research.evolve.islands import SEED_PROGRAM
from laplace_research.evolve.mutate import INTERFACE, Scored
from laplace_research.evolve.panel import Panel
from laplace_research.evolve.run import Run, RunConfig


def test_mutators_are_told_about_senses_and_events() -> None:
    assert "ctx.events" in INTERFACE
    assert "ctx.senses" in INTERFACE
    assert "ctx.peer_senses" in INTERFACE


def test_a_score_names_its_reference_and_where_it_gains() -> None:
    scored = Scored("code", 0.004, {"h1": 0.003, "h4": 0.005, "seg:shock": -0.002, "seg:releases": 0.011}, "EV-HAR")
    text = scored.describe()
    assert text.startswith("+0.0040 nats per question vs EV-HAR (h1 +0.0030, h4 +0.0050)")
    assert "where it gains and loses: shock -0.0020, releases +0.0110" in text


def test_a_run_can_measure_programs_against_ev_har(panel: Panel, tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    r2 = b.hourly_returns(panel.close) ** 2
    dvol = np.full(panel.close.shape, np.nan)
    dvol[0] = 100 * np.sqrt(b.trailing_mean(r2[0], 168) * b.HOURS_PER_YEAR)
    sensed = panel.with_senses({"rv": r2 * rng.uniform(0.5, 1.5, r2.shape), "dvol": dvol})
    run_dir = tmp_path / "evolve-t"
    run_dir.mkdir()
    sensed.save(run_dir / "panel.npz")
    config = RunConfig(
        data_start="2025-01-01T00:00",
        data_end="2025-02-20T00:00",
        search=("2025-01-31T00:00", "2025-02-10T00:00", 7),
        validation=("2025-02-10T00:00", "2025-02-15T00:00", 5),
        holdout=("2025-02-15T00:00", "2025-02-19T00:00", 5),
        reference="ev-har",
    )
    (run_dir / "config.json").write_text(json.dumps(asdict(config)))
    run = Run(run_dir)
    assert run.reference == "EV-HAR"
    assert (run_dir / "reference-ev-har.npz").exists()
    assert np.array_equal(Run(run_dir).search_reference, run.search_reference)  # read back from the cache
    with Evaluator(run_dir / "panel.npz", workers=1) as evaluator:
        score = run.score(evaluator, SEED_PROGRAM)
    assert score.error is None
    assert score.breakdown is not None
    assert {"seg:calm", "seg:shock", "seg:outer bins"} <= set(score.breakdown)
