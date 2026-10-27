"""Rolling-origin evaluation: every model refitted before each test period, scored on every hour of it.

A single holdout, looked at again and again, stops being a test. Here each fold has a test period (normally a
calendar month); the models are fitted on everything from the start of the data to the period before it, the learned
layer stops early on that period before, and the test period is scored at every hour, after a gap that keeps every
label of one window out of the next. Gains are paired per forecast hour (mean over assets).

Compared, per fold:
- GARCH(1,1)-t, the classical bar;
- EV-HAR (IV-HAR scaled for sessions, weekends and releases), the senses-aware bar;
- EV-NET (``learned.ev_net``), EV-HAR corrected by a small network: exactly what a release ships.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np

from laplace_research.data.events import Calendar
from laplace_research.evolve import learned
from laplace_research.evolve.baselines import ev_har, garch, iv_har
from laplace_research.evolve.context import LOOKBACK
from laplace_research.evolve.evaluate import MAX_HORIZON, log_scores, realized_bins
from laplace_research.evolve.feed import HORIZONS, FloatArray, IntArray, finalize
from laplace_research.evolve.panel import Panel
from laplace_research.scoring import circular_block_indices, interval

BLOCK_HOURS = 72
PAIRS = (("ev-net", "ev-har"), ("ev-har", "garch"), ("ev-net", "garch"))
SCOPES = {"1h": (1,), "4h": (4,), "24h": (24,), "1h+4h": (1, 4)}


def _hours(panel: Panel, start: datetime, end: datetime) -> IntArray:
    """Every forecast hour from ``start`` whose labels at every horizon settle before ``end``."""
    lo = max(panel.index_at_or_after(start), LOOKBACK - 1)
    hi = min(panel.index_at_or_after(end), len(panel.ts)) - MAX_HORIZON
    return np.arange(lo, hi, dtype=np.int64)


def _per_hour(probs: FloatArray, realized: IntArray) -> FloatArray:
    """``[horizon, hour]`` mean log score over assets."""
    out: FloatArray = log_scores(finalize(probs), realized).mean(axis=0)
    return out


def rolling(panel: Panel, calendar: Calendar, start: datetime, boundaries: Sequence[datetime]) -> dict[str, Any]:
    """Folds between consecutive ``boundaries``: fold k tests ``[boundaries[k], boundaries[k + 1])``, validates on
    ``[boundaries[k - 1], boundaries[k])`` and fits on ``[start, boundaries[k - 1])``."""
    pw = learned.Panelwise.of(panel.close, panel.high, panel.low, panel.senses, panel.assets)
    series: dict[tuple[str, str], list[FloatArray]] = {pair: [] for pair in PAIRS}
    folds = []
    for k in range(1, len(boundaries) - 1):
        fit_hours = _hours(panel, start, boundaries[k - 1])
        valid_hours = _hours(panel, boundaries[k - 1], boundaries[k])
        test_hours = _hours(panel, boundaries[k], boundaries[k + 1])
        fit_times = fit_hours[::2]
        realized_fit = realized_bins(panel, fit_times)
        g = garch(panel.close, realized_fit, fit_times, slice(int(fit_hours[0]), int(fit_hours[-1]) + 1))
        iv = iv_har(panel.close, panel.senses["rv"], panel.senses["dvol"], panel.assets, realized_fit, fit_times)
        ev = ev_har(iv, panel.close, panel.ts, calendar, fit_times)
        net = learned.ev_net(ev, pw, calendar, panel.ts, fit_hours, valid_hours, lambda t: realized_bins(panel, t))
        realized = realized_bins(panel, test_hours)
        scores = {
            "garch": _per_hour(g.probs(test_hours), realized),
            "ev-har": _per_hour(ev.probs(test_hours), realized),
            "ev-net": _per_hour(net.probs(test_hours), realized),
        }
        row: dict[str, Any] = {"test_start": boundaries[k].isoformat(), "hours": len(test_hours)}
        for a, b in PAIRS:
            diff = scores[a] - scores[b]
            series[(a, b)].append(diff)
            row[f"{a} vs {b}"] = {s: float(diff[[HORIZONS.index(h) for h in hs]].mean()) for s, hs in SCOPES.items()}
        folds.append(row)
    pooled: dict[str, Any] = {}
    for a, b in PAIRS:
        diff = np.concatenate(series[(a, b)], axis=1)
        indices = circular_block_indices(diff.shape[1], BLOCK_HOURS)
        out: dict[str, Any] = {}
        for scope, hs in SCOPES.items():
            per_hour = diff[[HORIZONS.index(h) for h in hs]].mean(axis=0).tolist()
            out[scope] = {"mean": float(np.mean(per_hour)), "interval95": interval(per_hour, indices)}
        out["folds_ahead_1h4h"] = sum(1 for f in folds if f[f"{a} vs {b}"]["1h+4h"] > 0)
        pooled[f"{a} vs {b}"] = out
    return {"folds": folds, "pooled": pooled, "hours": int(sum(f["hours"] for f in folds))}


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
    return "\n".join(lines)
