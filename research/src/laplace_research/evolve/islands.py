"""Seed hypotheses, one per island. Each steers the mutations on its island towards a different idea, which keeps
the population diverse; nothing stops an island from abandoning its hypothesis if reality disagrees."""

from __future__ import annotations

import tomllib
from pathlib import Path

SEED_PROGRAM = '''\
import math
import numpy as np


def forecast(ctx):
    """Seed: a zero-mean Laplace distribution scaled by the last week's realized volatility."""
    r = np.diff(np.log(ctx.close[-169:]))
    sd_hour = math.sqrt(float(np.mean(r * r)))
    return ctx.lib.laplace_probs(sd_hour * math.sqrt(ctx.horizon), ctx.edges)
'''

ISLANDS: dict[str, str] = {
    "clustering": (
        "Volatility clusters: large recent moves predict larger moves soon after, and the effect decays. Weight recent "
        "hours more than older ones (EWMA, GARCH-like recursions, several decay rates blended)."
    ),
    "seasonality": (
        "Volatility follows the clock: it depends on the hour of day (Asian, European and US sessions, the US open) "
        "and the day of week (quiet weekends). Estimate these patterns from the asset's own history in the window and "
        "scale the forecast for the hours the horizon will cover."
    ),
    "spillover": (
        "Volatility spills over between assets: BTC's and ETH's recent behavior leads the others, and market-wide "
        "volatility is a better estimate than one noisy asset. Use ctx.peers."
    ),
    "tails": (
        "The shape matters as much as the scale: tails are fatter than normal, fatter at short horizons than at 24 "
        "hours, and after crashes the downside tail grows. Choose the distribution family, its tail weight and any "
        "asymmetry from the recent data."
    ),
    "ranges": (
        "High-low ranges carry more information about volatility than closes alone. Use range-based estimators "
        "(Parkinson, Garman-Klass, Rogers-Satchell) and combine them with close-to-close estimates."
    ),
    "activity": (
        "Trading activity predicts volatility: abnormal volume relative to its own recent level signals that larger "
        "moves are coming; quiet volume signals calm."
    ),
    "horizons": (
        "Volatility mean-reverts: over 24 hours it moves toward a long-run level, over 1 hour the latest hours matter "
        "most. Combine short, medium and long windows with weights that depend on the horizon (HAR-like)."
    ),
    "regimes": (
        "Markets switch between calm and turbulent regimes. Detect the current regime (volatility of volatility, "
        "trend strength, recent jumps) and use a different forecast in each, or mix them by regime probability."
    ),
}


def read_islands(path: Path) -> dict[str, str]:
    """Island hypotheses from a TOML file: one ``name = "directive"`` per island (multi-line strings allowed), in file
    order. Line breaks and indentation inside a directive are folded into single spaces, as in ``ISLANDS``."""
    data = tomllib.loads(path.read_text())
    directives = {name: " ".join(str(text).split()) for name, text in data.items() if isinstance(text, str)}
    if not directives or len(directives) != len(data) or not all(directives.values()):
        raise ValueError(f"{path}: expected one non-empty directive string per island name")
    return directives
