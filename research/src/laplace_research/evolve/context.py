"""What a forecasting program sees: the past, and nothing else, except the times of scheduled events.

Contexts are only ever built by ``history.HistoryReader``, whose buffer holds no candle after the forecast time. The
one thing from the future a context holds is when events are scheduled (``data.events``), which their publishers
announce in advance; what they announce is never in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from laplace_research.data.events import Events
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
    ``events`` lists the scheduled events (``data.events``: ``fomc``, ``cpi``, ``nfp``, ``us-open``, ``us-close``) from
    24 hours before to 48 hours after the forecast time, as (hours from the forecast time, kind), oldest first; one
    with ``0 <= hours < horizon`` happens inside the forecast's window. It is empty where the calendar does not reach.
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
    events: Events = ()


def make_context(
    peers: dict[str, Candles],
    asset: str,
    horizon: int,
    candle_open_ts: int,
    peer_senses: dict[str, dict[str, FloatArray]] | None = None,
    events: Events = (),
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
        events=events,
    )
