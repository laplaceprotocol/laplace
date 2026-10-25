"""What a forecasting program sees: the past, and nothing else.

Contexts are only ever built by ``history.HistoryReader``, whose buffer holds no candle after the forecast time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from laplace_research.evolve.feed import FloatArray, edges
from laplace_research.evolve.lib import LIB, Lib

LOOKBACK = 720
HOUR = 3600


def _readonly(a: FloatArray) -> FloatArray:
    a.flags.writeable = False
    return a


EDGES = {h: _readonly(edges(h)) for h in (1, 4, 24)}


@dataclass(frozen=True, slots=True)
class Candles:
    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    volume: FloatArray


@dataclass(frozen=True, slots=True)
class Context:
    """Read-only inputs for one forecast.

    Arrays hold the last ``LOOKBACK`` hourly candles, oldest first; the last element is the candle that has just
    closed. ``peers`` holds the same window for every feed asset (including this one), aligned hour by hour.
    ``senses`` holds the asset's other data over the same hours (``evolve.senses``: ``dvol``, ``funding``, ``rv``;
    NaN where unknown) and ``peer_senses`` the same for every asset; both are empty when the panel has no senses.
    """

    asset: str
    horizon: int
    edges: FloatArray
    hour: int
    weekday: int
    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    volume: FloatArray
    peers: dict[str, Candles]
    lib: Lib
    senses: dict[str, FloatArray] = field(default_factory=dict)
    peer_senses: dict[str, dict[str, FloatArray]] = field(default_factory=dict)


def make_context(
    peers: dict[str, Candles],
    asset: str,
    horizon: int,
    candle_open_ts: int,
    peer_senses: dict[str, dict[str, FloatArray]] | None = None,
) -> Context:
    """Assemble a context for ``asset`` from per-asset windows ending with the candle that opened at
    ``candle_open_ts``."""
    closed_at = datetime.fromtimestamp(candle_open_ts + HOUR, UTC)
    own = peers[asset]
    return Context(
        asset=asset,
        horizon=horizon,
        edges=EDGES[horizon],
        hour=closed_at.hour,
        weekday=closed_at.weekday(),
        open=own.open,
        high=own.high,
        low=own.low,
        close=own.close,
        volume=own.volume,
        peers=peers,
        lib=LIB,
        senses=(peer_senses or {}).get(asset, {}),
        peer_senses=peer_senses or {},
    )
