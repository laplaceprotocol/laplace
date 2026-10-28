"""Measure an evolution's champions against opponents fitted on the same data, at every hour.

The search window decided which programs survived and the validation window chooses among them; the holdout decides
nothing. Both are scored at every hour: a sample every few hours sees only some hours of the day.

1. **Champions.** Each island's candidates are its best programs by search score. Each runs on every hour of the
   validation window, and the island's champion is the one that scores best there at 1 and 4 hours (the scope of the
   primary comparison and of the live rules). The run's candidate is the champion that scores best there.
2. **Opponents fitted on the same data** (``rolling.fit_models``): GARCH and EV-HAR on the search window, as the
   run's reference is; EV-NET and EV-GBM on the search window, stopped on the validation window. The run's seed
   program is scored too: every champion descends from it.
3. **Pools.** The champions with EV-HAR, and with EV-GBM, mixed with per-horizon weights fitted on the validation
   window by EM to a certified gap.
4. **The holdout,** scored once at every hour. Gains are paired per forecast hour (the mean over assets), at each
   horizon and at 1 and 4 hours pooled (the scope the live rules judge), with 95% intervals from a circular block
   bootstrap over 72-hour blocks (168-hour blocks as a sensitivity), and per calendar month. The primary comparison,
   fixed before any holdout score exists, is the candidate against EV-HAR at 1 and 4 hours; every other row is
   secondary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from laplace_research.data.events import load as load_calendar
from laplace_research.evolve import learned
from laplace_research.evolve.evaluate import Evaluator, Outcome, log_scores, realized_bins
from laplace_research.evolve.feed import HORIZONS, FloatArray, IntArray, finalize
from laplace_research.evolve.pool import fit_pool, mix, realized_probs
from laplace_research.evolve.rolling import fit_models, hours_between
from laplace_research.evolve.run import TIME_LIMIT_S, Run, parse_utc, segments
from laplace_research.scoring import circular_block_indices, interval

CANDIDATES = 3  # per island, by search score
CHUNK_HOURS = 500  # hours per worker: 10,500 program calls, about the size of one search evaluation
BLOCK_HOURS = (72, 168)
WINDOWS = ("validation", "holdout")
SCOPES = {"1h": (1,), "4h": (4,), "24h": (24,), "1h+4h": (1, 4), "all": HORIZONS}
JUDGED = [HORIZONS.index(h) for h in SCOPES["1h+4h"]]  # what champions are chosen on, and the primary's scope
PRIMARY = ("candidate", "ev-har")
COMPARISONS = (
    PRIMARY,
    ("candidate", "seed"),
    ("candidate", "ev-gbm"),
    ("candidate", "ev-net"),
    ("candidate", "garch"),
    ("seed", "ev-har"),
    ("pool(champions+ev-har)", "ev-har"),
    ("pool(champions+ev-gbm)", "ev-gbm"),
    ("ev-gbm", "ev-har"),
    ("ev-net", "ev-har"),
)

Forecasts = dict[str, FloatArray]  # window -> final forecasts [asset, horizon, hour, bin] at every hour of it


def submit_chunks(evaluator: Evaluator, code: str, hours: IntArray, chunk_hours: int) -> list[Future[Outcome]]:
    """Run ``code`` at ``hours`` in chunks, each in a fresh worker reading history forward."""
    return [
        evaluator.submit(code, hours[i : i + chunk_hours], TIME_LIMIT_S * 3) for i in range(0, len(hours), chunk_hours)
    ]


def collect(futures: Sequence[Future[Outcome]]) -> FloatArray | str:
    """A program's final forecasts over every chunk, or the first chunk's error."""
    parts = []
    for future in futures:
        try:
            outcome = future.result()
        except BrokenProcessPool:
            return "worker crashed"
        if outcome.probs is None:
            return outcome.error or "no forecasts"
        parts.append(finalize(outcome.probs))
    return np.concatenate(parts, axis=2)


def per_hour(final: FloatArray, realized: IntArray) -> FloatArray:
    """``[horizon, hour]`` log score, the mean over assets."""
    out: FloatArray = log_scores(final, realized).mean(axis=0)
    return out


def summary(diff: FloatArray) -> dict[str, Any]:
    """Mean gain of ``[horizon, hour]`` differences per scope, with 95% intervals for each block length."""
    out: dict[str, Any] = {"hours": int(diff.shape[1])}
    resamples = {b: circular_block_indices(diff.shape[1], min(b, diff.shape[1])) for b in BLOCK_HOURS}
    for scope, hs in SCOPES.items():
        series = diff[[HORIZONS.index(h) for h in hs]].mean(axis=0).tolist()
        out[scope] = {"mean": float(np.mean(series))} | {
            f"interval95_{b}h": interval(series, indices) for b, indices in resamples.items()
        }
    return out


def per_horizon_pool(members: Mapping[str, Forecasts], realized_val: IntArray) -> tuple[dict[str, Any], Forecasts]:
    """Members mixed with weights fitted per horizon on the validation window."""
    names = list(members)
    first = members[names[0]]
    pooled = {w: np.empty_like(first[w]) for w in WINDOWS}
    fits: dict[str, Any] = {}
    for j, h in enumerate(HORIZONS):
        p = np.stack([realized_probs(members[n]["validation"][:, j], realized_val[:, j]) for n in names])
        fit = fit_pool(p)
        fits[f"h{h}"] = {
            "weights": dict(zip(names, map(float, fit.weights), strict=True)),
            "gap": fit.gap,
            "iterations": fit.iterations,
        }
        for w in WINDOWS:
            pooled[w][:, j] = mix(fit.weights, [members[n][w][:, j] for n in names])
    return fits, pooled


def measure(
    run_dir: Path, workers: int = 8, candidates: int = CANDIDATES, chunk_hours: int = CHUNK_HOURS
) -> dict[str, Any]:
    run = Run(run_dir)
    panel, cfg = run.panel, run.config
    search_start, search_end = parse_utc(cfg.search[0]), parse_utc(cfg.search[1])
    val_start, val_end = parse_utc(cfg.validation[0]), parse_utc(cfg.validation[1])
    hold_start, hold_end = parse_utc(cfg.holdout[0]), parse_utc(cfg.holdout[1])
    if search_end != val_start or val_end != hold_start:
        raise ValueError("measure needs adjacent windows: search, then validation, then holdout")
    hours = {
        "validation": hours_between(panel, val_start, val_end),
        "holdout": hours_between(panel, hold_start, hold_end),
    }
    realized = {w: realized_bins(panel, h) for w, h in hours.items()}

    pw = learned.Panelwise.of(panel.close, panel.high, panel.low, panel.senses, panel.assets)
    models = fit_models(panel, load_calendar(), pw, search_start, val_start, hold_start)
    forecasts: dict[str, Forecasts] = {
        name: {w: finalize(model.probs(h)) for w, h in hours.items()} for name, model in models.items()
    }

    seed = next(p for p in run.db.all_programs() if p.mutator == "seed")
    islands = [(str(i["name"]), run.db.best(int(i["id"]), candidates)) for i in run.db.islands()]
    failures: dict[str, str] = {}
    champions: list[dict[str, Any]] = []
    with Evaluator(run_dir / "panel.npz", workers=workers) as evaluator:
        codes = {seed.code} | {p.code for _, best in islands for p in best}
        jobs = {code: submit_chunks(evaluator, code, hours["validation"], chunk_hours) for code in codes}
        validation = {code: collect(futures) for code, futures in jobs.items()}
        for island, best in islands:
            scored = []
            for p in best:
                out = validation[p.code]
                if isinstance(out, str):
                    failures[f"{island}#{p.id} on validation"] = out
                else:
                    scored.append((float(np.mean(log_scores(out, realized["validation"])[:, JUDGED])), p, out))
            if scored:
                score, p, out = max(scored, key=lambda s: s[0])
                champions.append(
                    {"name": f"{island}#{p.id}", "island": island, "program": p, "score": score, "forecasts": out}
                )
        if not champions:
            raise RuntimeError(f"no champion survived validation: {failures}")
        holdout_codes = {seed.code} | {c["program"].code for c in champions}
        jobs = {code: submit_chunks(evaluator, code, hours["holdout"], chunk_hours) for code in holdout_codes}
        holdout = {code: collect(futures) for code, futures in jobs.items()}

    seed_val, seed_hold = validation[seed.code], holdout[seed.code]
    if isinstance(seed_val, str) or isinstance(seed_hold, str):
        raise RuntimeError(f"the seed program failed: {seed_val if isinstance(seed_val, str) else seed_hold}")
    forecasts["seed"] = {"validation": seed_val, "holdout": seed_hold}
    candidate = max(champions, key=lambda c: c["score"])
    for c in champions:
        out = holdout[c["program"].code]
        if isinstance(out, str):
            failures[f"{c['name']} on the holdout"] = out
            continue
        forecasts[c["name"]] = {"validation": c["forecasts"], "holdout": out}
    scored_champions = [c["name"] for c in champions if c["name"] in forecasts]

    pools: dict[str, Any] = {}
    if candidate["name"] in forecasts:
        forecasts["candidate"] = forecasts[candidate["name"]]
        for opponent in ("ev-har", "ev-gbm"):
            label = f"pool(champions+{opponent})"
            members = {n: forecasts[n] for n in (*scored_champions, opponent)}
            pools[label], forecasts[label] = per_horizon_pool(members, realized["validation"])

    def diff(a: str, b: str, w: str) -> FloatArray:
        out: FloatArray = per_hour(forecasts[a][w], realized[w]) - per_hour(forecasts[b][w], realized[w])
        return out

    months = {
        w: np.array([datetime.fromtimestamp(int(panel.ts[h]) + 3600, UTC).strftime("%Y-%m") for h in hours[w]])
        for w in WINDOWS
    }
    comparisons: dict[str, Any] = {}
    for a, b in COMPARISONS:
        if a not in forecasts or b not in forecasts:
            continue
        row: dict[str, Any] = {}
        for w in WINDOWS:
            d = diff(a, b, w)
            short = d[JUDGED].mean(axis=0)
            row[w] = summary(d) | {
                "months_1h4h": {m: float(short[months[w] == m].mean()) for m in sorted(set(months[w]))}
            }
        comparisons[f"{a} vs {b}"] = row

    champion_rows = []
    for c in champions:
        row = {"name": c["name"], "island": c["island"], "search_gain": c["program"].gain}
        val = per_hour(c["forecasts"], realized["validation"]) - per_hour(
            forecasts["ev-har"]["validation"], realized["validation"]
        )
        row["validation"] = summary(val)
        if c["name"] in forecasts:
            row["holdout"] = summary(diff(c["name"], "ev-har", "holdout"))
        champion_rows.append(row)

    segment_gains: dict[str, Any] = {}
    if "candidate" in forecasts:
        gain = log_scores(forecasts["candidate"]["holdout"], realized["holdout"]) - log_scores(
            forecasts["ev-har"]["holdout"], realized["holdout"]
        )
        for name, mask in segments(panel, hours["holdout"], realized["holdout"]).items():
            m = mask[:, JUDGED]
            if m.any():
                segment_gains[name] = {"share": float(m.mean()), "mean": float(gain[:, JUDGED][m].mean())}

    primary = comparisons.get(f"{PRIMARY[0]} vs {PRIMARY[1]}", {}).get("holdout", {}).get("1h+4h")
    result: dict[str, Any] = {
        "run": run_dir.name,
        "config": json.loads((run_dir / "config.json").read_text()),
        "hours": {w: len(h) for w, h in hours.items()},
        "candidates_per_island": candidates,
        "seed_program_id": seed.id,
        "champions": champion_rows,
        "candidate": candidate["name"],
        "failures": failures,
        "pools": pools,
        "comparisons": comparisons,
        "candidate_vs_ev_har_by_segment_holdout_1h4h": segment_gains,
        "ev_gbm_rounds": models["ev-gbm"].params["gbm"]["rounds"],
        "primary": None
        if primary is None
        else {
            "comparison": f"{PRIMARY[0]} vs {PRIMARY[1]}, holdout, 1h+4h",
            **primary,
            "beats": primary["interval95_72h"][0] > 0,
        },
    }
    (run_dir / "measure.json").write_text(json.dumps(result, indent=1) + "\n")
    (run_dir / "measure.md").write_text(render(result))
    return result


def _cell(s: dict[str, Any], scope: str, block: int = 72) -> str:
    lo, hi = s[scope][f"interval95_{block}h"]
    return f"{s[scope]['mean']:+.4f} ({lo:+.4f} to {hi:+.4f})"


def render(r: dict[str, Any]) -> str:
    h = r["hours"]
    lines = [
        f"# Measured: `{r['run']}`",
        "",
        f"Every hour scored: {h['validation']} validation hours, {h['holdout']} holdout hours, 7 assets x 3 horizons. "
        f"Champions: per island, the best of its {r['candidates_per_island']} best programs by search score, chosen on "
        f"validation at 1h+4h; the candidate is the champion best there: **{r['candidate']}**. Opponents are fitted on "
        "the same data (search window; EV-NET and EV-GBM stopped on validation). Gains in nats per question, paired "
        "per forecast hour; 95% intervals from a circular block bootstrap over 72-hour blocks.",
        "",
    ]
    if r["primary"] is not None:
        p = r["primary"]
        lines += [
            f"**Primary ({p['comparison']}):** {p['mean']:+.4f}, 72h blocks {p['interval95_72h'][0]:+.4f} to "
            f"{p['interval95_72h'][1]:+.4f}, 168h blocks {p['interval95_168h'][0]:+.4f} to "
            f"{p['interval95_168h'][1]:+.4f}: " + ("beats EV-HAR." if p["beats"] else "does not beat EV-HAR."),
            "",
        ]
    lines += [
        "| comparison | window | 1h | 4h | 24h | 1h+4h | all |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, row in r["comparisons"].items():
        for w in WINDOWS:
            cells = " | ".join(_cell(row[w], s) for s in SCOPES)
            lines.append(f"| {name} | {w} | {cells} |")
    lines += ["", "1h+4h by month (validation months chose the champions; holdout months chose nothing):", ""]
    months = sorted({m for row in r["comparisons"].values() for w in WINDOWS for m in row[w]["months_1h4h"]})
    lines += ["| comparison | " + " | ".join(months) + " |", "|---|" + "---|" * len(months)]
    for name, row in r["comparisons"].items():
        by_month = row["validation"]["months_1h4h"] | row["holdout"]["months_1h4h"]
        lines.append(f"| {name} | " + " | ".join(f"{by_month[m]:+.4f}" if m in by_month else "" for m in months) + " |")
    lines += [
        "",
        "Champions against EV-HAR, 1h+4h:",
        "",
        "| champion | search gain (all horizons) | validation | holdout |",
        "|---|---:|---|---|",
    ]
    for c in r["champions"]:
        val = _cell(c["validation"], "1h+4h")
        hold = _cell(c["holdout"], "1h+4h") if "holdout" in c else "failed"
        lines.append(f"| {c['name']} | {c['search_gain']:+.4f} | {val} | {hold} |")
    if r["candidate_vs_ev_har_by_segment_holdout_1h4h"]:
        lines += ["", "Candidate against EV-HAR on the holdout by segment, 1h+4h:", ""]
        lines += ["| segment | share of questions | gain |", "|---|---:|---:|"]
        for name, s in r["candidate_vs_ev_har_by_segment_holdout_1h4h"].items():
            lines.append(f"| {name} | {s['share']:.0%} | {s['mean']:+.4f} |")
    for label, fits in r["pools"].items():
        lines += ["", f"**{label}** weights (fitted on validation, per horizon):", ""]
        names = list(next(iter(fits.values()))["weights"])
        lines += ["| member | " + " | ".join(fits) + " |", "|---|" + "---:|" * len(fits)]
        for n in names:
            lines.append(f"| {n} | " + " | ".join(f"{f['weights'][n]:.3f}" for f in fits.values()) + " |")
    if r["failures"]:
        lines += ["", "**Failures:** " + "; ".join(f"{k}: {v}" for k, v in r["failures"].items())]
    return "\n".join(lines) + "\n"
