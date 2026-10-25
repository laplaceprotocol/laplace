"""Past-only access to the candle panel for running untrusted programs.

The process that runs a program never loads a candle from after the forecast time. The panel is written once as a
time-major binary file (hour, asset, field) next to a metadata file holding only the assets and timestamps. A reader
allocates a zero-filled buffer and reads the file forward as forecast times advance, so every array a program can
reach, including a view's ``.base``, contains the past and zeros, never the future. Forecast times must therefore be
visited in non-decreasing order, and each evaluation runs in a fresh process. Contexts also list the scheduled events
around their forecast time, from a calendar of publication times (``data.events``) that holds no market data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from laplace_research.data import events as calendar_data
from laplace_research.evolve.context import HOUR, LOOKBACK, Candles, Context, make_context
from laplace_research.evolve.feed import FloatArray, IntArray
from laplace_research.evolve.panel import Panel, senses_path

FIELDS = ("open", "high", "low", "close", "volume")


def history_paths(panel_path: Path) -> tuple[Path, Path]:
    """Where the time-major data file and its metadata live, next to the panel."""
    return panel_path.with_name(panel_path.stem + ".history.bin"), panel_path.with_name(panel_path.stem + ".meta.npz")


def write_history(panel: Panel, panel_path: Path) -> tuple[Path, Path]:
    """Candle fields first, then the panel's senses in name order, for every hour and asset."""
    data_path, meta_path = history_paths(panel_path)
    names = sorted(panel.senses)
    rows = np.stack([getattr(panel, f) for f in FIELDS] + [panel.senses[n] for n in names], axis=-1)
    data_path.write_bytes(np.ascontiguousarray(rows.transpose(1, 0, 2), dtype="<f8").tobytes())  # [hour, asset, field]
    np.savez(meta_path, assets=np.array(panel.assets), ts=panel.ts, senses=np.array(names, dtype=str))
    return data_path, meta_path


def _sense_names(meta_path: Path) -> list[str]:
    with np.load(meta_path) as meta:
        return [str(n) for n in meta["senses"]] if "senses" in meta.files else []


def ensure_history(panel_path: Path) -> tuple[Path, Path]:
    """The history files, rewritten if missing or if the panel's senses changed since they were written."""
    data_path, meta_path = history_paths(panel_path)
    expected: list[str] = []
    if senses_path(panel_path).exists():
        with np.load(senses_path(panel_path)) as data:
            expected = sorted(data.files)
    if not (data_path.exists() and meta_path.exists()) or _sense_names(meta_path) != expected:
        write_history(Panel.load(panel_path), panel_path)
    return data_path, meta_path


class HistoryReader:
    """Reads candles forward in time; contexts are views into a buffer that holds nothing beyond the last hour read."""

    def __init__(self, data_path: Path, meta_path: Path, calendar: calendar_data.Calendar | None = None) -> None:
        with np.load(meta_path) as meta:
            self.assets: tuple[str, ...] = tuple(str(a) for a in meta["assets"])
            self.ts: IntArray = meta["ts"].astype(np.int64)
            self.senses: tuple[str, ...] = tuple(str(n) for n in meta["senses"]) if "senses" in meta.files else ()
        width = len(FIELDS) + len(self.senses)
        self._data_path = data_path
        self._buffer: FloatArray = np.zeros((len(self.ts), len(self.assets), width))
        self._buffer.flags.writeable = False
        self._row_bytes = len(self.assets) * width * 8
        self._loaded = 0
        self._calendar = calendar if calendar is not None else calendar_data.load()
        self._events: tuple[int, calendar_data.Events] = (-1, ())

    @property
    def loaded(self) -> int:
        """Number of hours read so far; hours ``>= loaded`` are zeros in the buffer."""
        return self._loaded

    def advance_to(self, t: int) -> None:
        """Make hours ``0..t`` available, reading nothing after hour ``t``."""
        if t + 1 <= self._loaded:
            return
        if t >= len(self.ts):
            raise IndexError(f"hour {t} is beyond the panel")
        self._buffer.flags.writeable = True
        try:
            view = memoryview(self._buffer).cast("B")[self._loaded * self._row_bytes : (t + 1) * self._row_bytes]
            with self._data_path.open("rb") as fh:
                fh.seek(self._loaded * self._row_bytes)
                if fh.readinto(view) != len(view):
                    raise OSError("history file is shorter than its metadata says")
        finally:
            self._buffer.flags.writeable = False
        self._loaded = t + 1

    def context(self, asset_index: int, t: int, horizon: int) -> Context:
        if t >= self._loaded:
            raise ValueError(f"hour {t} has not been read yet; call advance_to first")
        if t + 1 < LOOKBACK:
            raise ValueError("not enough history for a context")
        window = self._buffer[t + 1 - LOOKBACK : t + 1]  # [hour, asset, field]
        peers = {name: Candles(*(window[:, i, k] for k in range(len(FIELDS)))) for i, name in enumerate(self.assets)}
        senses = {
            name: {sense: window[:, i, len(FIELDS) + j] for j, sense in enumerate(self.senses)}
            for i, name in enumerate(self.assets)
        }
        if self._events[0] != t:  # the same for every asset and horizon at one forecast time
            self._events = (t, self._calendar.around(int(self.ts[t]) + HOUR))
        return make_context(peers, self.assets[asset_index], horizon, int(self.ts[t]), senses, self._events[1])
