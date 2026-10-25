"""How much hourly variance moves around scheduled events (``data.events``), and what knowing them is worth.

``event_study``: the candle that contains each event, the one before it and the three after it, each compared with
weekday candles at the same UTC hours (events move with daylight saving, so those hours differ through the year) that
are not at that offset from an event; for releases, days with any release are left out of the comparison. Ratios are of
mean squared hourly log returns, per asset, then averaged over assets; intervals resample the events.

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
DAY = 86_400
OFFSETS = (-1, 0, 1, 2, 3)  # candles from the one before an event to the third after it
BLOCK_HOURS = 72
N_BOOT = 2000
SCOPES = {"1h": (1,), "4h": (4,), "24h": (24,), "1h+4h": (1, 4), "all": HORIZONS}


def _containing(panel: Panel, times: IntArray) -> IntArray:
    """Index of the candle that contains each instant, for instants with a full candle on either side."""
    index: IntArray = np.searchsorted(panel.ts, times, side="right") - 1
    return index[(index >= -min(OFFSETS)) & (index < len(panel.ts) - max(OFFSETS))]


def event_study(panel: Panel, calendar: Calendar, *, seed: int = 0) -> dict[str, Any]:
    r2 = hourly_returns(panel.close) ** 2  # r2[a, t]: the squared return of the candle that opens at ts[t]
    hour = (panel.ts // HOUR) % 24
    day = panel.ts // DAY
    weekday = (day + 3) % 7 < 5  # 1970-01-01 was a Thursday
    kinds = np.array(calendar.kinds)
    release_days = np.isin(day, np.unique(calendar.times[np.isin(kinds, RELEASES)] // DAY))
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"assets": list(panel.assets), "first": int(panel.ts[0]), "last": int(panel.ts[-1])}
    for kind in (*RELEASES, "us-open"):
        index = _containing(panel, calendar.times[kinds == kind])
        if not len(index):
            continue
        rows = {}
        for k in OFFSETS:
            at = index + k
            control = weekday & np.isin(hour, np.unique(hour[at]))
            control[at] = False
            if kind != "us-open":
                control &= ~release_days
            reference = r2[:, control].mean(axis=1)
            per_asset = r2[:, at].mean(axis=1) / reference
            draws = rng.integers(0, len(at), (N_BOOT, len(at)))
            boot = (r2[:, at][:, draws].mean(axis=2) / reference[:, None]).mean(axis=0)
            rows[str(k)] = {
                "ratio": float(per_asset.mean()),
                "interval95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                "per_asset": per_asset.tolist(),
            }
        out[kind] = {"n": len(index), "by_offset": rows}
    return out


def render(study: dict[str, Any]) -> str:
    names = {-1: "hour before", 0: "event hour", 1: "+1 h", 2: "+2 h", 3: "+3 h"}
    lines = [
        "| event | n | " + " | ".join(names[k] for k in OFFSETS) + " |",
        "|---|---|" + "---|" * len(OFFSETS),
    ]
    for kind in (*RELEASES, "us-open"):
        if kind not in study:
            continue
        row = study[kind]
        cells = []
        for k in OFFSETS:
            r = row["by_offset"][str(k)]
            cells.append(f"{r['ratio']:.2f}x ({r['interval95'][0]:.2f} to {r['interval95'][1]:.2f})")
        lines.append(f"| {kind} | {row['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def overlay_screen(panel: Panel, config: RunConfig, calendar: Calendar) -> dict[str, Any]:
    """EV-HAR's gain over IV-HAR in nats per question, per scope, with 95% intervals from a circular block bootstrap
    over hours, and the overlay's fitted coefficients."""
    fit = config.fit_times(panel).times
    realized_fit = realized_bins(panel, fit)
    iv = iv_har(panel.close, panel.senses["rv"], panel.senses["dvol"], panel.assets, realized_fit, fit)
    ev = ev_har(iv, panel.close, panel.ts, calendar, fit)
    start = panel.index_at_or_after(parse_utc(config.validation[0]))
    end = min(panel.index_at_or_after(parse_utc(config.holdout[1])), len(panel.ts)) - max(HORIZONS)
    every = np.arange(start, end)
    realized = realized_bins(panel, every)
    diff = log_scores(finalize(ev.probs(every)), realized) - log_scores(finalize(iv.probs(every)), realized)
    indices = circular_block_indices(len(every), BLOCK_HOURS)
    gains = {}
    for scope, horizons in SCOPES.items():
        series = diff[:, [HORIZONS.index(h) for h in horizons]].mean(axis=(0, 1)).tolist()
        gains[scope] = {"mean": float(np.mean(series)), "interval95": interval(series, indices)}
    forecast_hour = (panel.ts[every] // HOUR + 1) % 24
    by_hour = {str(h): float(diff[:, 0, forecast_hour == h].mean()) for h in range(24)}
    return {
        "first": int(panel.ts[start]) + HOUR,
        "hours": len(every),
        "ev_har_vs_iv_har": gains,
        "one_hour_gain_by_utc_hour": by_hour,
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
