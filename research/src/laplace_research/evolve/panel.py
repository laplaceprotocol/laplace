"""An aligned, read-only panel of hourly candles for every feed asset, with optional senses (other data aligned to
the same hours, see ``evolve.senses``), kept in a separate file next to the panel."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from laplace_research.data.candles import HOUR, fetch_hourly
from laplace_research.evolve.feed import ASSETS, FloatArray, IntArray, product

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class Panel:
    """``ts[t]`` is the open time of hour ``t``; price arrays are ``[asset, t]`` and refer to that hour's candle.

    A forecast "at index t" is made at the close of candle ``t``, i.e. at ``ts[t] + HOUR``.
    """

    assets: tuple[str, ...]
    ts: IntArray
    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    volume: FloatArray
    senses: dict[str, FloatArray] = field(default_factory=dict)  # name -> [asset, t], NaN where unknown

    def __post_init__(self) -> None:
        for name in FIELDS:
            getattr(self, name).flags.writeable = False
        self.ts.flags.writeable = False
        for name, values in self.senses.items():
            if values.shape != self.close.shape:
                raise ValueError(f"sense {name} is not aligned to the panel")
            values.flags.writeable = False

    def with_senses(self, senses: dict[str, FloatArray]) -> Panel:
        return replace(self, senses=senses)

    def index_at_or_after(self, when: datetime) -> int:
        return int(np.searchsorted(self.ts, int(when.timestamp())))

    def save(self, path: Path) -> None:
        np.savez_compressed(path, assets=np.array(self.assets), ts=self.ts, **{f: getattr(self, f) for f in FIELDS})
        if self.senses:
            arrays: dict[str, Any] = dict(self.senses)  # a **dict of arrays reads to the stubs like allow_pickle
            np.savez_compressed(senses_path(path), **arrays)

    @classmethod
    def load(cls, path: Path) -> Panel:
        senses: dict[str, FloatArray] = {}
        if senses_path(path).exists():
            with np.load(senses_path(path)) as data:
                senses = {name: data[name].astype(np.float64) for name in data.files}
        with np.load(path) as data:
            candles: dict[str, Any] = {f: data[f].astype(np.float64) for f in FIELDS}
            return cls(
                assets=tuple(str(a) for a in data["assets"]),
                ts=data["ts"].astype(np.int64),
                senses=senses,
                **candles,
            )


def senses_path(panel_path: Path) -> Path:
    """Where a panel's senses are kept: a file next to it, so a panel's own file never changes when senses are added."""
    return panel_path.with_name(panel_path.stem + ".senses.npz")


def build(start: datetime, end: datetime, cache_dir: Path, *, assets: tuple[str, ...] = ASSETS) -> Panel:
    """Fetch every asset and keep the hours all of them share."""
    series = {}
    with httpx.Client(timeout=30.0, headers={"User-Agent": "laplace-research"}) as client:
        for asset in assets:
            series[asset] = {
                c.ts: c
                for c in fetch_hourly(
                    product(asset), start, end, client=client, cache_dir=cache_dir, max_filled_hours=48
                )
            }
    common = sorted(set.intersection(*(set(s) for s in series.values())))
    gaps = int(np.count_nonzero(np.diff(common) != HOUR))
    if gaps:
        raise ValueError(f"assets do not share a contiguous range of hours ({gaps} breaks)")
    arrays: dict[str, Any] = {
        f: np.array([[getattr(series[a][ts], f) for ts in common] for a in assets]) for f in FIELDS
    }
    return Panel(assets=assets, ts=np.array(common, dtype=np.int64), **arrays)


def load_or_build(path: Path, start: datetime, end: datetime, cache_dir: Path) -> Panel:
    if path.exists():
        return Panel.load(path)
    panel = build(start, end, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
    return panel


def utc(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, UTC)
