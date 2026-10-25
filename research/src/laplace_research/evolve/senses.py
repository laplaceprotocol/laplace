"""Senses aligned to a panel's hours.

For hour t (the candle that opens at ``ts[t]``), every value is known when that candle closes, at the forecast time,
so the past-only history reader can serve senses exactly like candles. Unavailable values are NaN.

- ``dvol``: the asset's 30-day implied volatility index (annualized percent) at the end of hour t; BTC and ETH only.
- ``rv``: the hour's realized variance from 5-minute log returns.
- ``funding``: the asset's perpetual funding rate at its last settlement at or before the end of hour t.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path

import httpx
import numpy as np

from laplace_research.data import senses as sources
from laplace_research.evolve.feed import FloatArray, IntArray, product
from laplace_research.evolve.panel import Panel, utc

NAMES = ("dvol", "funding", "rv")
HOUR = 3600
FUNDING_LOOKBACK = timedelta(hours=8)


def funding_symbol(asset: str) -> str:
    return f"{asset}USDT"


def carry_forward(settlements: Sequence[tuple[int, float]], times: IntArray) -> FloatArray:
    """At each time, the last settled value at or before it (NaN before the first settlement)."""
    out = np.full(len(times), np.nan)
    if not settlements:
        return out
    stamps = np.array([t for t, _ in settlements], dtype=np.int64)
    values = np.array([v for _, v in settlements], dtype=np.float64)
    index = np.searchsorted(stamps, times, side="right") - 1
    known = index >= 0
    out[known] = values[index[known]]
    return out


def build(
    panel: Panel, cache_dir: Path, *, client: httpx.Client | None = None, problems: list[str] | None = None
) -> dict[str, FloatArray]:
    """Every sense for every asset and hour of ``panel``.

    With ``problems``, a source that fails for an asset leaves that sense unknown (NaN) and is reported there instead
    of raising: the live engine must seal its hour whatever a third-party API does.
    """
    start, end = utc(int(panel.ts[0])), utc(int(panel.ts[-1]) + HOUR)
    out = {name: np.full((len(panel.assets), len(panel.ts)), np.nan) for name in NAMES}
    own_client = client is None
    client = client or httpx.Client(timeout=30.0, headers={"User-Agent": "laplace-research"})

    def attempt(name: str, asset: str, fill: Callable[[], None]) -> None:
        if problems is None:
            fill()
            return
        try:
            fill()
        except Exception as exc:
            problems.append(f"{name} for {asset}: {type(exc).__name__}: {exc}"[:200])

    try:
        for i, asset in enumerate(panel.assets):

            def rv(i: int = i, asset: str = asset) -> None:
                values = sources.realized_hourly(product(asset), start, end, client=client, cache_dir=cache_dir)
                out["rv"][i] = [values.get(int(t), np.nan) for t in panel.ts]

            def dvol(i: int = i, asset: str = asset) -> None:
                values = sources.dvol_hourly(asset, start, end, client=client, cache_dir=cache_dir)
                out["dvol"][i] = [values.get(int(t), np.nan) for t in panel.ts]

            def funding(i: int = i, asset: str = asset) -> None:
                symbol = funding_symbol(asset)
                rates = sources.funding_rates(symbol, start - FUNDING_LOOKBACK, end, client=client, cache_dir=cache_dir)
                out["funding"][i] = carry_forward(rates, panel.ts + HOUR)

            attempt("rv", asset, rv)
            if asset in sources.DVOL_CURRENCIES:
                attempt("dvol", asset, dvol)
            attempt("funding", asset, funding)
    finally:
        if own_client:
            client.close()
    return out
