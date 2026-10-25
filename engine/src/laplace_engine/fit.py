"""Fitting new pools for a release on a research run, and judging them on its holdout as development evidence.

Every program runs on the run's validation and holdout windows in the sandboxed evaluator. Pool weights are fitted
on validation by EM to a certified optimality gap, either one set for all horizons or one set per horizon. The
holdout comparison against GARCH and HAR uses the same per-issue means and block bootstrap as the research reports.
The holdout has been inspected before, so this only screens candidates: their live record is the test.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from laplace_engine.quantize import WEIGHT_SCALE
from laplace_engine.release import Program, Weights, weights_ppm
from laplace_research.evolve.evaluate import Evaluator, log_scores, realized_bins
from laplace_research.evolve.feed import HORIZONS, FloatArray, IntArray, finalize
from laplace_research.evolve.pool import fit_pool, realized_probs
from laplace_research.evolve.run import Run
from laplace_research.scoring import circular_block_indices, interval

TIME_LIMIT_S = 600
BLOCK_HOURS = 72
SCOPES: dict[str, tuple[int, ...]] = {"1h": (1,), "4h": (4,), "24h": (24,), "1h+4h": (1, 4), "all": HORIZONS}


@dataclass(frozen=True)
class Windows:
    validation: dict[str, FloatArray]  # final forecasts [asset, horizon, sample, bin]
    holdout: dict[str, FloatArray]
    realized_validation: IntArray  # [asset, horizon, sample]
    realized_holdout: IntArray
    holdout_step_hours: int


def run_windows(run_dir: Path, programs: Sequence[Program], *, workers: int) -> Windows:
    """Final forecasts of every program on the run's validation and holdout windows."""
    run = Run(run_dir)
    config, panel = run.config, run.panel
    validation, holdout = config.split(panel, "validation"), config.split(panel, "holdout")
    finals: dict[str, dict[str, FloatArray]] = {"validation": {}, "holdout": {}}
    with Evaluator(run_dir / "panel.npz", workers=workers) as evaluator:
        futures = {
            (p.name, split.name): evaluator.submit(p.source, split.times, TIME_LIMIT_S)
            for p in programs
            for split in (validation, holdout)
        }
        for (name, split_name), future in futures.items():
            outcome = future.result()
            if outcome.probs is None:
                raise RuntimeError(f"{name} failed on {split_name}: {outcome.error}")
            finals[split_name][name] = finalize(outcome.probs)
    return Windows(
        finals["validation"],
        finals["holdout"],
        realized_bins(panel, validation.times),
        realized_bins(panel, holdout.times),
        int(config.holdout[2]),
    )


def fit_weights(finals: Mapping[str, FloatArray], realized: IntArray, horizons: Sequence[int]) -> Weights:
    """EM weights for the given members over the questions at ``horizons``, as parts per million."""
    columns = [HORIZONS.index(h) for h in horizons]
    names = list(finals)
    p = np.stack([realized_probs(finals[n][:, columns], realized[:, columns]) for n in names])
    fit = fit_pool(p)
    return weights_ppm(dict(zip(names, map(float, fit.weights), strict=True)))


def mix(finals: Mapping[str, FloatArray], weights_at: Callable[[int], Weights]) -> FloatArray:
    """The pooled final forecast, horizon by horizon."""
    any_member = next(iter(finals.values()))
    out = np.zeros_like(any_member)
    for j, h in enumerate(HORIZONS):
        for name, w in weights_at(h):
            out[:, j] += w / WEIGHT_SCALE * finals[name][:, j]
    return out


def compare(
    windows: Windows, candidates: Mapping[str, FloatArray], opponents: Sequence[str] = ("garch", "har")
) -> dict[str, dict[str, dict[str, Any]]]:
    """Holdout gains of each candidate over each opponent, per scope: mean and 95% interval over issues."""
    realized = windows.realized_holdout
    indices = circular_block_indices(realized.shape[2], max(1, round(BLOCK_HOURS / windows.holdout_step_hours)))
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for name, forecasts in candidates.items():
        own = log_scores(forecasts, realized)
        out[name] = {}
        for opponent in opponents:
            diff = own - log_scores(windows.holdout[opponent], realized)
            row = {}
            for scope, horizons in SCOPES.items():
                columns = [HORIZONS.index(h) for h in horizons]
                series = diff[:, columns].mean(axis=(0, 1)).tolist()
                row[scope] = {"mean": float(np.mean(series)), "interval95": interval(series, indices)}
            out[name][opponent] = row
    return out


def sense_baselines(run_dir: Path) -> list[Program]:
    """RV-HAR and IV-HAR as programs, with parameters fitted on the run's search window (needs a panel with senses)."""
    from laplace_engine.baselines import ladder_programs
    from laplace_research.evolve.baselines import ladder
    from laplace_research.evolve.run import parse_utc

    run = Run(run_dir)
    config, panel = run.config, run.panel
    if "rv" not in panel.senses:
        raise ValueError(f"{run_dir.name}'s panel has no senses: run `laplace-research evolve senses` first")
    fit = config.fit_times(panel)
    start, end, _ = config.search
    fit_range = slice(panel.index_at_or_after(parse_utc(start)), panel.index_at_or_after(parse_utc(end)))
    rungs = ladder(
        panel.close, realized_bins(panel, fit.times), fit.times, fit_range, senses=panel.senses, assets=panel.assets
    )
    where = f"{run_dir.name}'s search window ({start[:10]} to {end[:10]})"
    sources = ladder_programs(rungs, panel.assets, where)
    return [Program(n, sources[n], True, f"baseline fitted on {where}") for n in ("rv-har", "iv-har") if n in sources]
