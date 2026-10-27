"""Rolling-origin evaluation: every model refitted before each test period, scored on every hour of it.

A single holdout, looked at again and again, stops being a test. Here each fold has a test period (normally a
calendar month); the models are fitted on everything from the start of the data to the period before it, the learned
layers stop early on that period before, and the test period is scored at every hour, after a gap that keeps every
label of one window out of the next. Gains are paired per forecast hour (mean over assets).

Compared, per fold:
- GARCH(1,1)-t, the classical bar;
- EV-HAR (IV-HAR scaled for sessions, weekends and releases), the senses-aware bar;
- EV-NET (``learned.ev_net``), EV-HAR corrected by a small network;
- EV-GBM (``gbm.ev_gbm``), EV-HAR corrected by boosted trees (needs the ``gbm`` extra).

With ``ages``, each fold's models also score every later test period, which shows how a fit fades: a model fitted
for one month against the same model refitted for a later month, on that later month (stale minus fresh), and each
learned layer against the EV-HAR of its own fit, by the number of months since the fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np

from laplace_research.data.events import Calendar
from laplace_research.evolve import gbm, learned
from laplace_research.evolve.baselines import ev_har, garch, iv_har
from laplace_research.evolve.context import LOOKBACK
from laplace_research.evolve.evaluate import MAX_HORIZON, log_scores, realized_bins
from laplace_research.evolve.feed import HORIZONS, FloatArray, IntArray, finalize
from laplace_research.evolve.panel import Panel
from laplace_research.scoring import circular_block_indices, interval

BLOCK_HOURS = 72
MODELS = ("garch", "ev-har", "ev-net", "ev-gbm")
PAIRS = (
    ("ev-gbm", "ev-har"),
    ("ev-net", "ev-har"),
    ("ev-gbm", "ev-net"),
    ("ev-har", "garch"),
    ("ev-gbm", "garch"),
)
SAME_FIT = (("ev-gbm", "ev-har"), ("ev-net", "ev-har"), ("ev-har", "garch"))  # by age, both from one fold's fit
SCOPES = {"1h": (1,), "4h": (4,), "24h": (24,), "1h+4h": (1, 4)}


def hours_between(panel: Panel, start: datetime, end: datetime) -> IntArray:
    """Every forecast hour from ``start`` whose labels at every horizon settle before ``end``."""
    lo = max(panel.index_at_or_after(start), LOOKBACK - 1)
    hi = min(panel.index_at_or_after(end), len(panel.ts)) - MAX_HORIZON
    return np.arange(lo, hi, dtype=np.int64)


def _per_hour(probs: FloatArray, realized: IntArray) -> FloatArray:
    """``[horizon, hour]`` mean log score over assets."""
    out: FloatArray = log_scores(finalize(probs), realized).mean(axis=0)
    return out


def _fit(
    panel: Panel, calendar: Calendar, pw: learned.Panelwise, start: datetime, valid_from: datetime, test_from: datetime
) -> dict[str, Any]:
    """Every model fitted on ``[start, valid_from)``, the learned layers stopped on ``[valid_from, test_from)``."""
    fit_hours = hours_between(panel, start, valid_from)
    valid_hours = hours_between(panel, valid_from, test_from)
    fit_times = fit_hours[::2]
    realized_fit = realized_bins(panel, fit_times)
    g = garch(panel.close, realized_fit, fit_times, slice(int(fit_hours[0]), int(fit_hours[-1]) + 1))
    iv = iv_har(panel.close, panel.senses["rv"], panel.senses["dvol"], panel.assets, realized_fit, fit_times)
    ev = ev_har(iv, panel.close, panel.ts, calendar, fit_times)

    def realized(t: IntArray) -> IntArray:
        return realized_bins(panel, t)

    return {
        "garch": g,
        "ev-har": ev,
        "ev-net": learned.ev_net(ev, pw, calendar, panel.ts, fit_hours, valid_hours, realized),
        "ev-gbm": gbm.ev_gbm(ev, pw, calendar, panel.ts, fit_hours, valid_hours, realized),
    }


def _summary(series: list[FloatArray]) -> dict[str, Any]:
    """Mean and 95% interval over every hour of ``series`` (``[horizon, hour]`` gains), per scope."""
    diff = np.concatenate(series, axis=1)
    indices = circular_block_indices(diff.shape[1], min(BLOCK_HOURS, diff.shape[1]))
    out: dict[str, Any] = {}
    for scope, hs in SCOPES.items():
        per_hour = diff[[HORIZONS.index(h) for h in hs]].mean(axis=0).tolist()
        out[scope] = {"mean": float(np.mean(per_hour)), "interval95": interval(per_hour, indices)}
    out["hours"] = int(diff.shape[1])
    return out


def rolling(
    panel: Panel, calendar: Calendar, start: datetime, boundaries: Sequence[datetime], *, ages: bool = False
) -> dict[str, Any]:
    """Folds between consecutive ``boundaries``: fold k tests ``[boundaries[k], boundaries[k + 1])``, validates on
    ``[boundaries[k - 1], boundaries[k])`` and fits on ``[start, boundaries[k - 1])``."""
    pw = learned.Panelwise.of(panel.close, panel.high, panel.low, panel.senses, panel.assets)
    periods = [hours_between(panel, boundaries[k], boundaries[k + 1]) for k in range(len(boundaries) - 1)]
    realized = [realized_bins(panel, hours) for hours in periods]
    # scores[k][m][model]: fold k's model on period m (m == k is the fold's own test; later ones only with ``ages``)
    scores: dict[int, dict[int, dict[str, FloatArray]]] = {}
    rounds = []
    for k in range(1, len(boundaries) - 1):
        models = _fit(panel, calendar, pw, start, boundaries[k - 1], boundaries[k])
        rounds.append({"test_start": boundaries[k].isoformat(), **models["ev-gbm"].params["gbm"]["rounds"]})
        scores[k] = {
            m: {name: _per_hour(model.probs(periods[m]), realized[m]) for name, model in models.items()}
            for m in (range(k, len(periods)) if ages else (k,))
        }
    folds = []
    for k in scores:
        row: dict[str, Any] = {"test_start": boundaries[k].isoformat(), "hours": len(periods[k])}
        for a, b in PAIRS:
            diff = scores[k][k][a] - scores[k][k][b]
            row[f"{a} vs {b}"] = {s: float(diff[[HORIZONS.index(h) for h in hs]].mean()) for s, hs in SCOPES.items()}
        folds.append(row)
    pooled = {}
    for a, b in PAIRS:
        pooled[f"{a} vs {b}"] = _summary([scores[k][k][a] - scores[k][k][b] for k in scores])
        pooled[f"{a} vs {b}"]["folds_ahead_1h4h"] = sum(1 for f in folds if f[f"{a} vs {b}"]["1h+4h"] > 0)
    result: dict[str, Any] = {"folds": folds, "pooled": pooled, "hours": sum(f["hours"] for f in folds)}
    result["ev_gbm_rounds"] = rounds
    if ages:
        result["ages"] = _ages(scores)
    return result


def _ages(scores: dict[int, dict[int, dict[str, FloatArray]]]) -> dict[str, Any]:
    """Gains by months since the fit: each model against its own refit for the later month (stale minus fresh), and
    each learned layer against the EV-HAR fitted with it."""
    stale: dict[str, dict[int, list[FloatArray]]] = {name: {} for name in MODELS}
    same: dict[str, dict[int, list[FloatArray]]] = {f"{a} vs {b}": {} for a, b in SAME_FIT}
    for k, by_period in scores.items():
        for m, by_model in by_period.items():
            age = m - k
            if age > 0 and m in scores:
                for name in MODELS:
                    stale[name].setdefault(age, []).append(by_model[name] - scores[m][m][name])
            for a, b in SAME_FIT:
                same[f"{a} vs {b}"].setdefault(age, []).append(by_model[a] - by_model[b])
    return {
        "stale minus fresh": {n: {str(a): _summary(s) for a, s in sorted(by.items())} for n, by in stale.items()},
        "same fit": {p: {str(a): _summary(s) for a, s in sorted(by.items())} for p, by in same.items()},
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "| fold (test from) | hours | " + " | ".join(f"{a} vs {b}, 1h+4h" for a, b in PAIRS) + " |",
        "|---|---|" + "---|" * len(PAIRS),
    ]
    for f in result["folds"]:
        cells = " | ".join(f"{f[f'{a} vs {b}']['1h+4h']:+.4f}" for a, b in PAIRS)
        lines.append(f"| {f['test_start'][:10]} | {f['hours']} | {cells} |")
    lines += ["", "| pooled | " + " | ".join(SCOPES) + " | folds ahead (1h+4h) |", "|---|" + "---|" * (len(SCOPES) + 1)]
    for pair, row in result["pooled"].items():
        cells = " | ".join(
            f"{row[s]['mean']:+.4f} ({row[s]['interval95'][0]:+.4f} to {row[s]['interval95'][1]:+.4f})" for s in SCOPES
        )
        lines.append(f"| {pair} | {cells} | {row['folds_ahead_1h4h']}/{len(result['folds'])} |")
    for title, table in result.get("ages", {}).items():
        lines += ["", f"| {title}, 1h+4h, by months since the fit | " + " | ".join(table) + " |"]
        lines.append("|---|" + "---|" * len(table))
        ages = sorted({int(a) for by in table.values() for a in by})
        for age in ages:
            row_cells = []
            for by in table.values():
                s = by.get(str(age))
                row_cells.append(
                    f"{s['1h+4h']['mean']:+.4f} ({s['1h+4h']['interval95'][0]:+.4f} to "
                    f"{s['1h+4h']['interval95'][1]:+.4f})"
                    if s
                    else ""
                )
            lines.append(f"| {age} | " + " | ".join(row_cells) + " |")
    return "\n".join(lines)
