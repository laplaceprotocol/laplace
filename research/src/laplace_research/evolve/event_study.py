"""How much hourly variance moves around scheduled events (``data.events``), and what knowing them is worth.

``event_study``: for each release kind, the candle that contains the release is compared with weekday candles at the
same UTC hours (the hours its releases fall in, which move with daylight saving) that contain no release of that kind;
the hour before and the three hours after are compared with all hours. NYSE opens are compared with every other weekday
candle. Ratios are of mean squared hourly log returns.

``overlay_screen``: EV-HAR against IV-HAR, both fitted on a run's search window, scored at every hour from the start of
validation to the end of the holdout. The run's own windows sample every few hours, which for 1-hour questions leaves
whole hours of the day unscored.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from laplace_research.data.events import RELEASES, Calendar
from laplace_research.evolve.baselines import ev_har, hourly_returns, iv_har
from laplace_research.evolve.evaluate import log_scores, realized_bins
from laplace_research.evolve.feed import HORIZONS, IntArray, finalize
from laplace_research.evolve.panel import Panel
from laplace_research.evolve.run import RunConfig, parse_utc
from laplace_research.scoring import circular_block_indices, interval

HOUR = 3600
AFTER = (1, 2, 3)
BLOCK_HOURS = 72
SCOPES = {"1h": (1,), "4h": (4,), "24h": (24,), "1h+4h": (1, 4), "all": HORIZONS}


def _containing(panel: Panel, times: IntArray) -> IntArray:
    """Index of the candle that contains each instant, for instants with a full candle on either side."""
    index: IntArray = np.searchsorted(panel.ts, times, side="right") - 1
    return index[(index >= 1) & (index < len(panel.ts) - max(AFTER))]


def event_study(panel: Panel, calendar: Calendar) -> dict[str, Any]:
    r2 = hourly_returns(panel.close) ** 2  # r2[a, t]: the squared return of the candle that opens at ts[t]
    hour = (panel.ts // HOUR) % 24
    weekday = ((panel.ts // 86400) + 3) % 7 < 5
    kinds = np.array(calendar.kinds)
    overall = r2[:, 1:].mean(axis=1)
    out: dict[str, Any] = {"assets": list(panel.assets), "first": int(panel.ts[0]), "last": int(panel.ts[-1])}
    for kind in (*RELEASES, "us-open"):
        index = _containing(panel, calendar.times[kinds == kind])
        if not len(index):
            continue
        marked = np.zeros(len(panel.ts), dtype=bool)
        marked[index] = True
        control = weekday & ~marked
        if kind != "us-open":
            control &= np.isin(hour, np.unique(hour[index]))  # the hours its releases fall in
        out[kind] = {
            "n": len(index),
            "event_hour": (r2[:, index].mean(axis=1) / r2[:, control].mean(axis=1)).tolist(),
            "hour_before": float(np.mean(r2[:, index - 1].mean(axis=1) / overall)),
            "hours_after": [float(np.mean(r2[:, index + k].mean(axis=1) / overall)) for k in AFTER],
        }
    return out


def render(study: dict[str, Any]) -> str:
    lines = [
        "| event | n | event hour (range over assets) | hour before | next hours |",
        "|---|---|---|---|---|",
    ]
    for kind in (*RELEASES, "us-open"):
        if kind not in study:
            continue
        row = study[kind]
        low, high = min(row["event_hour"]), max(row["event_hour"])
        after = ", ".join(f"{x:.2f}x" for x in row["hours_after"])
        lines.append(f"| {kind} | {row['n']} | {low:.2f}x to {high:.2f}x | {row['hour_before']:.2f}x | {after} |")
    return "\n".join(lines)


def overlay_screen(panel: Panel, config: RunConfig, calendar: Calendar) -> dict[str, Any]:
    """EV-HAR's gain over IV-HAR in nats per question, per scope, with 95% intervals from a circular block bootstrap
    over hours, and the overlay's fitted coefficients."""
    fit = config.fit_times(panel).times
    realized_fit = realized_bins(panel, fit)
    iv = iv_har(panel.close, panel.senses["rv"], panel.senses["dvol"], panel.assets, realized_fit, fit)
    ev = ev_har(iv, panel.close, panel.ts, calendar, fit)
    start = panel.index_at_or_after(parse_utc(config.validation[0]))
    end = min(panel.index_at_or_after(parse_utc(config.holdout[1])), len(panel.ts) - max(HORIZONS) - 1)
    every = np.arange(start, end)
    realized = realized_bins(panel, every)
    diff = log_scores(finalize(ev.probs(every)), realized) - log_scores(finalize(iv.probs(every)), realized)
    indices = circular_block_indices(len(every), BLOCK_HOURS)
    gains = {}
    for scope, horizons in SCOPES.items():
        series = diff[:, [HORIZONS.index(h) for h in horizons]].mean(axis=(0, 1)).tolist()
        gains[scope] = {"mean": float(np.mean(series)), "interval95": interval(series, indices)}
    return {
        "first": int(panel.ts[start]) + HOUR,
        "hours": len(every),
        "ev_har_vs_iv_har": gains,
        "overlay": ev.params["overlay"],
        "nu": {str(h): nu for h, nu in ev.nu.items()},
    }


def render_screen(screen: dict[str, Any]) -> str:
    cells = " | ".join(
        f"{r['mean']:+.4f} ({r['interval95'][0]:+.4f} to {r['interval95'][1]:+.4f})"
        for r in screen["ev_har_vs_iv_har"].values()
    )
    names = " | ".join(screen["ev_har_vs_iv_har"])
    lines = [f"| hours | {names} |", "|---" * (len(screen["ev_har_vs_iv_har"]) + 1) + "|"]
    lines.append(f"| {screen['hours']} | {cells} |")
    effects = ["", "| input | 1h | 4h | 24h |", "|---|---|---|---|"]
    by_horizon = {h: dict(zip(o["inputs"], o["beta"], strict=True)) for h, o in screen["overlay"].items()}
    for name in ("weekend", "releases", "recent-releases", "upcoming-releases", "us-open", "us-session"):
        row = " | ".join(f"{np.exp(by_horizon[str(h)][name]):.2f}x" for h in HORIZONS)
        effects.append(f"| {name} | {row} |")
    return "\n".join(lines + effects)
