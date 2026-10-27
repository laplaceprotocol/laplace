"""Run programs over sampled forecast times and score them with proper scoring rules.

Each evaluation runs in a fresh, spawned worker process (nothing inherited from the parent, which holds the full
panel) with a CPU-time alarm and an address-space limit. The worker reads candles forward through
``history.HistoryReader``, so when a program forecasts at hour ``t`` its process has never loaded a candle after ``t``.
The program is called once per (asset, horizon, time); any exception, timeout or malformed output rejects the whole
program with a message that the next mutation prompt can show to the model.
"""

from __future__ import annotations

import os
import resource
import signal
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from laplace_research.data import events
from laplace_research.evolve import params as parameter_files
from laplace_research.evolve.context import LOOKBACK
from laplace_research.evolve.feed import HORIZONS, N_BINS, FloatArray, IntArray, bin_index
from laplace_research.evolve.history import HistoryReader, ensure_history
from laplace_research.evolve.panel import Panel
from laplace_research.evolve.sandbox import ProgramRejected, load

MAX_HORIZON = max(HORIZONS)
MEMORY_LIMIT_BYTES = 3 * 1024**3


@dataclass(frozen=True)
class Split:
    name: str
    times: IntArray  # panel hour indices at whose close forecasts are made


def sample_times(panel: Panel, start: datetime, end: datetime, step_hours: int) -> IntArray:
    """Every ``step_hours``-th hour from ``start`` with a full lookback, keeping only forecasts whose labels at every
    horizon settle before ``end``.

    A forecast at hour ``t`` settles at the close of hour ``t + h``. Requiring ``t + MAX_HORIZON < index(end)`` means
    every label of a window is known strictly before the next window's first forecast, so nothing a window is
    fitted or selected on comes from the window after it.
    """
    lo = max(panel.index_at_or_after(start), LOOKBACK - 1)
    hi = min(panel.index_at_or_after(end), len(panel.ts)) - MAX_HORIZON
    return np.arange(lo, hi, step_hours, dtype=np.int64)


def realized_bins(panel: Panel, times: IntArray) -> IntArray:
    """``[asset, horizon, sample]`` bin index of the realized log return."""
    out = np.empty((len(panel.assets), len(HORIZONS), len(times)), dtype=np.int64)
    for j, h in enumerate(HORIZONS):
        log_returns = np.log(panel.close[:, times + h] / panel.close[:, times])
        out[:, j, :] = bin_index(log_returns, h)
    return out


def log_scores(final: FloatArray, realized: IntArray) -> FloatArray:
    """Log probability that final forecasts gave the realized bin: ``[asset, horizon, sample]``."""
    out: FloatArray = np.log(np.take_along_axis(final, realized[..., None], axis=-1)[..., 0])
    return out


def ranked_probability_scores(final: FloatArray, realized: IntArray) -> FloatArray:
    """Ranked probability score of final forecasts (lower is better), which rewards mass near the realized bin."""
    cdf = np.cumsum(final, axis=-1)
    observed = (np.arange(N_BINS) >= realized[..., None]).astype(np.float64)
    out: FloatArray = np.sum((cdf - observed) ** 2, axis=-1) / (N_BINS - 1)
    return out


def day_groups(panel: Panel, times: IntArray) -> list[str]:
    return [datetime.fromtimestamp(int(panel.ts[t]), UTC).strftime("%Y-%m-%d") for t in times]


# ---- worker side -------------------------------------------------------------------------------------------------

_HISTORY: tuple[Path, Path] | None = None
_CALENDAR: Path | None = None  # the scheduled events contexts list; None lists none


class _Timeout(Exception):
    pass


def _on_alarm(signum: int, frame: Any) -> None:
    raise _Timeout


def _init_worker(data_path: str, meta_path: str, calendar_path: str) -> None:
    global _HISTORY, _CALENDAR
    os.environ["OMP_NUM_THREADS"] = "1"
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    signal.signal(signal.SIGALRM, _on_alarm)
    np.seterr(all="ignore")
    # Warnings raised inside numpy while running a program would otherwise go through the warnings machinery from the
    # program's restricted frame; the program's output is validated anyway.
    warnings.simplefilter("ignore")
    _HISTORY = (Path(data_path), Path(meta_path))
    _CALENDAR = Path(calendar_path) if calendar_path else None


@dataclass(frozen=True)
class Outcome:
    probs: FloatArray | None  # [asset, horizon, sample, bin], raw (finalize before scoring)
    error: str | None
    seconds: float


def _run(source: str, times: IntArray, time_limit_s: int, params: bytes | None = None) -> Outcome:
    assert _HISTORY is not None
    if np.any(np.diff(times) < 0):
        raise ValueError("forecast times must be non-decreasing")
    started = time.perf_counter()
    try:
        fn = load(source)
    except ProgramRejected as exc:
        return Outcome(None, f"rejected: {exc}", 0.0)
    try:
        calendar = events.load(_CALENDAR) if _CALENDAR is not None else events.EMPTY
    except Exception as exc:  # a broken calendar fails this evaluation, not the process that runs it
        return Outcome(None, f"calendar: {type(exc).__name__}: {exc}"[:400], 0.0)
    try:
        arrays = parameter_files.decode(params) if params is not None else parameter_files.EMPTY
    except Exception as exc:  # a broken parameter file fails its program, not the process or the hour
        return Outcome(None, f"params: {type(exc).__name__}: {exc}"[:400], 0.0)
    reader = HistoryReader(*_HISTORY, calendar=calendar)
    out = np.empty((len(reader.assets), len(HORIZONS), len(times), N_BINS))
    signal.alarm(time_limit_s)
    try:
        for n, t in enumerate(times):
            reader.advance_to(int(t))
            for a in range(len(reader.assets)):
                for j, h in enumerate(HORIZONS):
                    raw = np.asarray(fn(reader.context(a, int(t), h, arrays)), dtype=np.float64)
                    if raw.shape != (N_BINS,) or not np.all(np.isfinite(raw)) or np.any(raw < 0) or raw.sum() <= 0:
                        return Outcome(None, f"bad output {np.array2string(raw, precision=3)[:200]}", 0.0)
                    out[a, j, n] = raw
    except _Timeout:
        return Outcome(None, f"timed out after {time_limit_s}s", float(time_limit_s))
    except MemoryError:
        return Outcome(None, "ran out of memory", time.perf_counter() - started)
    except Exception as exc:  # the program's own failure, reported back to the next mutation
        return Outcome(None, f"{type(exc).__name__}: {exc}"[:400], time.perf_counter() - started)
    finally:
        signal.alarm(0)
    return Outcome(out, None, time.perf_counter() - started)


class Evaluator:
    """A pool of spawned worker processes; each evaluation gets a fresh process that reads history forward."""

    def __init__(self, panel_path: Path, workers: int, *, calendar: Path | None = events.DEFAULT) -> None:
        """``calendar``: the scheduled events contexts list (``data.events``); None lists none."""
        data_path, meta_path = ensure_history(panel_path)
        self._history: tuple[str, str, str] = (str(data_path), str(meta_path), str(calendar) if calendar else "")
        self._workers = workers
        self._pool = self._new_pool()

    def _new_pool(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self._workers,
            mp_context=get_context("spawn"),
            initializer=_init_worker,
            initargs=self._history,
            max_tasks_per_child=1,
        )

    def submit(self, source: str, times: IntArray, time_limit_s: int, params: bytes | None = None) -> Any:
        """Run ``source`` at ``times``; ``params`` is its parameter file, if it has one."""
        try:
            return self._pool.submit(_run, source, times, time_limit_s, params)
        except BrokenProcessPool:
            self._pool = self._new_pool()
            return self._pool.submit(_run, source, times, time_limit_s, params)

    def run(self, source: str, times: IntArray, time_limit_s: int) -> Outcome:
        try:
            result: Outcome = self.submit(source, times, time_limit_s).result()
        except BrokenProcessPool:
            self._pool = self._new_pool()
            return Outcome(None, "worker crashed", 0.0)
        return result

    def close(self) -> None:
        self._pool.shutdown(cancel_futures=True)

    def __enter__(self) -> Evaluator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def mean_gain(log_p: FloatArray, reference_log_p: FloatArray) -> float:
    """Mean log-likelihood gain over the reference, in nats per question."""
    return float(np.mean(log_p - reference_log_p))


def breakdown(log_p: FloatArray, reference_log_p: FloatArray, assets: Sequence[str]) -> dict[str, float]:
    gain = log_p - reference_log_p
    out = {f"h{h}": float(np.mean(gain[:, j])) for j, h in enumerate(HORIZONS)}
    out |= {a: float(np.mean(gain[i])) for i, a in enumerate(assets)}
    return out
