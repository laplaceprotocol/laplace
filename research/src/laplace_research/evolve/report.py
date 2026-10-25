"""Judge an evolution honestly.

1. Each island's champion is chosen on the validation window from its three best programs by search score, so the
   search score never decides on its own. Window boundaries are purged: every label a window is fitted or selected on
   settles before the next window's first forecast.
2. Champions, the baseline ladder and two pools are then scored once on the holdout window, which played no part in
   any choice. Every forecast is finalized exactly once; a pool mixes final forecasts and is not smoothed again.
3. The pools ("the demon") are linear pools whose weights are fitted on validation by EM, run until a certified
   optimality gap. Each member's renormalized leave-one-out contribution is reported as a diagnostic, not as a payout
   rule.
4. Intervals come from a circular block bootstrap over forecast issues (each issue weighted equally), with the same
   resamples for every comparison and a sensitivity table over block lengths.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np

from laplace_research.evolve.baselines import ladder
from laplace_research.evolve.db import Program
from laplace_research.evolve.evaluate import (
    Evaluator,
    day_groups,
    log_scores,
    ranked_probability_scores,
    realized_bins,
)
from laplace_research.evolve.feed import HORIZONS, FloatArray, IntArray, finalize
from laplace_research.evolve.pool import fit_pool, mix, realized_probs, renormalized_contributions
from laplace_research.evolve.run import TIME_LIMIT_S, Run, parse_utc
from laplace_research.experiments.report import spearman
from laplace_research.scoring import circular_block_indices, interval

BLOCK_HOURS = 72
SENSITIVITY_BLOCK_HOURS = (24, 48, 72, 168)


@dataclass
class Member:
    name: str
    kind: str  # "champion", "baseline" or "pool"
    search_gain: float | None
    validation: FloatArray  # final forecasts [asset, horizon, sample, bin]
    holdout: FloatArray
    program: Program | None = None


@dataclass
class Pool:
    weights: dict[str, float]
    gap: float
    iterations: int
    contribution_holdout: dict[str, float]
    contribution_validation: dict[str, float]
    validation: FloatArray
    holdout: FloatArray


def linear_pool(members: Sequence[Member], realized_val: IntArray, realized_hold: IntArray) -> Pool:
    pv = np.stack([realized_probs(m.validation, realized_val) for m in members])
    ph = np.stack([realized_probs(m.holdout, realized_hold) for m in members])
    fit = fit_pool(pv)
    names = [m.name for m in members]
    return Pool(
        weights=dict(zip(names, map(float, fit.weights), strict=True)),
        gap=fit.gap,
        iterations=fit.iterations,
        contribution_holdout=dict(zip(names, map(float, renormalized_contributions(fit.weights, ph)), strict=True)),
        contribution_validation=dict(zip(names, map(float, renormalized_contributions(fit.weights, pv)), strict=True)),
        validation=mix(fit.weights, [m.validation for m in members]),
        holdout=mix(fit.weights, [m.holdout for m in members]),
    )


def per_issue(diff: FloatArray) -> FloatArray:
    """Average ``[asset, horizon, sample]`` over assets and horizons, leaving one value per forecast issue."""
    out: FloatArray = diff.mean(axis=(0, 1))
    return out


def report(run_dir: Path, workers: int = 8) -> dict[str, Any]:
    run = Run(run_dir)
    panel, cfg = run.panel, run.config
    val, hold, fit = cfg.split(panel, "validation"), cfg.split(panel, "holdout"), cfg.fit_times(panel)
    rv, rh = realized_bins(panel, val.times), realized_bins(panel, hold.times)
    days = day_groups(panel, hold.times)
    step = int(cfg.holdout[2])

    start, end, _ = cfg.search
    fit_range = slice(panel.index_at_or_after(parse_utc(start)), panel.index_at_or_after(parse_utc(end)))
    members = [
        Member(b.name, "baseline", None, finalize(b.probs(val.times)), finalize(b.probs(hold.times)))
        for b in ladder(
            panel.close,
            realized_bins(panel, fit.times),
            fit.times,
            fit_range,
            senses=panel.senses,
            assets=panel.assets,
            ts=panel.ts,
        )
    ]
    naive_val, naive_hold = log_scores(members[0].validation, rv), log_scores(members[0].holdout, rh)

    failures: dict[str, list[str]] = {}
    with Evaluator(run_dir / "panel.npz", workers=workers) as evaluator:
        for island in run.db.islands():
            candidates = run.db.best(int(island["id"]), 3)
            futures = [evaluator.submit(p.code, val.times, TIME_LIMIT_S * 3) for p in candidates]
            best: tuple[float, Program, FloatArray] | None = None
            for program, future in zip(candidates, futures, strict=True):
                outcome = future.result()
                if outcome.probs is None:
                    failures.setdefault(island["name"], []).append(f"#{program.id} validation: {outcome.error}")
                    continue
                final = finalize(outcome.probs)
                gain = float(np.mean(log_scores(final, rv) - naive_val))
                if best is None or gain > best[0]:
                    best = (gain, program, final)
            if best is None:
                continue
            _, program, val_final = best
            hold_outcome = evaluator.run(program.code, hold.times, TIME_LIMIT_S * 3)
            if hold_outcome.probs is None:
                failures.setdefault(island["name"], []).append(f"#{program.id} holdout: {hold_outcome.error}")
                continue
            members.append(
                Member(
                    f"{island['name']}#{program.id}",
                    "champion",
                    program.gain,
                    val_final,
                    finalize(hold_outcome.probs),
                    program,
                )
            )

    champions = [m for m in members if m.kind == "champion"]
    by_name = {m.name: m for m in members}
    garch = next(m for m in members if m.name.startswith("garch"))
    har = next(m for m in members if m.name.startswith("har"))

    pools: dict[str, Pool] = {}
    for label, names in (
        ("pool(evolved)", [m.name for m in champions]),
        ("pool(evolved+classical)", [m.name for m in members if m.name != "naive"]),
    ):
        if len(names) >= 2:
            pools[label] = linear_pool([by_name[n] for n in names], rv, rh)
            members.append(Member(label, "pool", None, pools[label].validation, pools[label].holdout))

    indices = circular_block_indices(len(hold.times), max(1, round(BLOCK_HOURS / step)))
    garch_h, har_h = log_scores(garch.holdout, rh), log_scores(har.holdout, rh)
    rows: list[dict[str, Any]] = []
    for m in members:
        lh = log_scores(m.holdout, rh)
        rows.append(
            {
                "name": m.name,
                "kind": m.kind,
                "search_gain": m.search_gain,
                "validation_gain": float(np.mean(log_scores(m.validation, rv) - naive_val)),
                "holdout_gain": float(np.mean(lh - naive_hold)),
                "vs_garch": float(np.mean(lh - garch_h)),
                "vs_garch_ci": interval(per_issue(lh - garch_h).tolist(), indices),
                "vs_har": float(np.mean(lh - har_h)),
                "vs_har_ci": interval(per_issue(lh - har_h).tolist(), indices),
                "holdout_by_horizon": {
                    f"h{h}": float(np.mean((lh - naive_hold)[:, j])) for j, h in enumerate(HORIZONS)
                },
                "rps": float(np.mean(ranked_probability_scores(m.holdout, rh))),
            }
        )
    rows.sort(key=lambda r: -r["holdout_gain"])

    headline = {}
    if "pool(evolved)" in pools:
        pooled = log_scores(pools["pool(evolved)"].holdout, rh)
        for name, reference in (("pool vs GARCH", garch_h), ("pool vs HAR", har_h)):
            series = per_issue(pooled - reference).tolist()
            daily: dict[str, list[float]] = {}
            for d, v in zip(days, series, strict=True):
                daily.setdefault(d, []).append(v)
            headline[name] = {
                "mean": fmean(series),
                "sd_per_issue": pstdev(series),
                "sd_daily_mean": pstdev(fmean(v) for v in daily.values()),
                "intervals_by_block_hours": {
                    hours: interval(series, circular_block_indices(len(series), max(1, round(hours / step))))
                    for hours in SENSITIVITY_BLOCK_HOURS
                },
                # Secondary: the pre-declared comparison is over all horizons; these are not corrected for three looks.
                "by_horizon": {
                    f"h{h}": {
                        "mean": float(np.mean((pooled - reference)[:, j])),
                        "interval": interval((pooled - reference)[:, j].mean(axis=0).tolist(), indices),
                    }
                    for j, h in enumerate(HORIZONS)
                },
            }

    best_single = max((r for r in rows if r["kind"] == "champion"), key=lambda r: r["validation_gain"], default=None)
    persistence = None
    if "pool(evolved)" in pools and len(champions) >= 3:
        p = pools["pool(evolved)"]
        names = list(p.contribution_holdout)
        persistence = spearman(
            [p.contribution_validation[n] for n in names], [p.contribution_holdout[n] for n in names]
        )

    programs = run.db.all_programs()
    summary: dict[str, Any] = {
        "run": run_dir.name,
        "config": json.loads((run_dir / "config.json").read_text()),
        "samples": {
            "validation": len(val.times),
            "holdout": len(hold.times),
            "holdout_days": len(set(days)),
            "holdout_step_hours": step,
        },
        "programs": run.db.meta(),
        "unrecorded_billed_replies": sum(
            1
            for p in programs
            if p.status == "failed"
            and p.cost_usd == 0
            and (p.error or "").startswith("mutation:")
            and "duplicate" not in (p.error or "")
        ),
        "rows": rows,
        "evaluation_failures": failures,
        "pools": {
            label: {
                "weights": p.weights,
                "gap": p.gap,
                "iterations": p.iterations,
                "contribution_holdout": p.contribution_holdout,
                "contribution_validation": p.contribution_validation,
            }
            for label, p in pools.items()
        },
        "headline": headline,
        "block_hours": BLOCK_HOURS,
        "best_single_by_validation": best_single["name"] if best_single else None,
        "contribution_persistence_spearman": persistence,
        "mutators": mutator_stats(programs),
    }
    (run_dir / "report.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "report.md").write_text(render(summary))
    champions_dir = run_dir / "champions"
    champions_dir.mkdir(exist_ok=True)
    for m in champions:
        assert m.program is not None
        (champions_dir / f"{m.name.replace('#', '-')}.py").write_text(m.program.code)
    return summary


def mutator_stats(programs: Sequence[Program]) -> dict[str, dict[str, float]]:
    by_id = {p.id: p for p in programs}
    out: dict[str, dict[str, float]] = {}
    for name in sorted({p.mutator for p in programs if p.mutator != "seed"}):
        mine = [p for p in programs if p.mutator == name]
        valid = [p for p in mine if p.status == "ok"]
        deltas = [
            p.gain - by_id[p.parent_id].gain  # type: ignore[operator]
            for p in valid
            if p.parent_id is not None and by_id[p.parent_id].gain is not None and p.gain is not None
        ]
        cost = sum(p.cost_usd for p in mine)
        improved = sum(d > 0 for d in deltas)
        out[name] = {
            "calls": len(mine),
            "valid_share": len(valid) / len(mine) if mine else 0.0,
            "improved_share": improved / len(mine) if mine else 0.0,
            "mean_delta": fmean(deltas) if deltas else 0.0,
            "cost_usd": cost,
            "usd_per_improvement": cost / improved if improved else float("inf"),
        }
    return out


def render(s: dict[str, Any]) -> str:
    n = s["samples"]
    lines = [
        f"# Evolution run `{s['run']}`",
        "",
        f"{s['programs']['programs']} programs ({s['programs']['valid']} valid), ${s['programs']['cost_usd']:.2f} of "
        f"recorded mutation calls ({s['unrecorded_billed_replies']} unusable replies have no recorded cost). Holdout: "
        f"{n['holdout']} forecast issues (every {n['holdout_step_hours']}h) over {n['holdout_days']} days x 7 assets "
        f"x 3 horizons; validation: {n['validation']} issues.",
        "Gains are mean log-likelihood gains of final forecasts in nats per question (positive = better). Champions "
        "were chosen on validation, window boundaries are purged, and the holdout played no part in any choice. 95% "
        f"intervals: circular block bootstrap over issues, {s['block_hours']}h blocks, issues weighted equally.",
        "",
        "| forecaster | kind | search | validation | holdout | vs GARCH | 95% CI | vs HAR | 95% CI | 1h | 4h | 24h |",
        "|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|",
    ]
    for r in s["rows"]:
        sg = "" if r["search_gain"] is None else f"{r['search_gain']:+.4f}"
        g, gh = r["vs_garch_ci"], r["vs_har_ci"]
        hb = r["holdout_by_horizon"]
        lines.append(
            f"| {r['name']} | {r['kind']} | {sg} | {r['validation_gain']:+.4f} | {r['holdout_gain']:+.4f} | "
            f"{r['vs_garch']:+.4f} | {g[0]:+.4f}…{g[1]:+.4f} | {r['vs_har']:+.4f} | {gh[0]:+.4f}…{gh[1]:+.4f} | "
            f"{hb['h1']:+.4f} | {hb['h4']:+.4f} | {hb['h24']:+.4f} |"
        )
    for island, errors in s["evaluation_failures"].items():
        lines += ["", f"**Not scored ({island}):** " + "; ".join(errors)]
    if s["headline"]:
        lines += ["", "Interval sensitivity to the block length (pool of evolved champions):", ""]
        hours = SENSITIVITY_BLOCK_HOURS
        lines += ["| comparison | mean | " + " | ".join(f"{h}h blocks" for h in hours) + " |"]
        lines += ["|---|---:|" + "---|" * len(hours)]
        for name, h in s["headline"].items():
            cis = " | ".join(f"{lo:+.4f}…{hi:+.4f}" for lo, hi in (h["intervals_by_block_hours"][k] for k in hours))
            lines.append(f"| {name} | {h['mean']:+.4f} | {cis} |")
        lines += ["", f"By horizon ({s['block_hours']}h blocks; secondary, not corrected for three comparisons):", ""]
        lines += ["| comparison | " + " | ".join(f"{h}h" for h in HORIZONS) + " |", "|---|" + "---|" * len(HORIZONS)]
        for name, h in s["headline"].items():
            cells = (h["by_horizon"][f"h{k}"] for k in HORIZONS)
            row = " | ".join(f"{c['mean']:+.4f} ({c['interval'][0]:+.4f}…{c['interval'][1]:+.4f})" for c in cells)
            lines.append(f"| {name} | {row} |")
    for label, p in s["pools"].items():
        lines += [
            "",
            f"**{label}**: weights fitted on validation (EM optimality gap {p['gap']:.1e} after {p['iterations']} "
            "iterations) and renormalized leave-one-out contributions (a diagnostic, not a payout rule):",
            "",
            "| member | weight | holdout contribution | validation contribution |",
            "|---|---:|---:|---:|",
        ]
        for name, w in sorted(p["weights"].items(), key=lambda kv: -kv[1]):
            hold, val = p["contribution_holdout"][name], p["contribution_validation"][name]
            lines.append(f"| {name} | {w:.3f} | {hold:+.5f} | {val:+.5f} |")
    lines += [
        "",
        "Contribution persistence (Spearman, validation vs holdout, evolved pool): "
        f"{s['contribution_persistence_spearman']}",
        "",
        "| mutator | calls | valid | improved on parent | mean change | recorded cost | $ per improvement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in s["mutators"].items():
        lines.append(
            f"| {name} | {m['calls']} | {m['valid_share']:.0%} | {m['improved_share']:.0%} | {m['mean_delta']:+.4f} | "
            f"${m['cost_usd']:.2f} | ${m['usd_per_improvement']:.3f} |"
        )
    return "\n".join(lines) + "\n"
