"""A fast circular block bootstrap for the mean of a time series.

Same resampling scheme as ``laplace_research.scoring`` (each resample concatenates ceil(n / block) blocks of
consecutive observations starting at uniform random positions, wrapping around the end, truncated to n; percentile
interval of the resampled means), computed from block sums over prefix sums instead of materialized index sets, so it
stays fast for a record that grows by an issue every hour.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

N_BOOT = 2_000


def block_starts(n: int, block: int, *, n_boot: int = N_BOOT, seed: int = 0) -> npt.NDArray[np.int64]:
    """Random block start positions, ``[n_boot, ceil(n / block)]``."""
    if not 1 <= block <= n:
        raise ValueError("block length must be between 1 and the number of observations")
    starts: npt.NDArray[np.int64] = np.random.default_rng(seed).integers(0, n, size=(n_boot, -(-n // block)))
    return starts


def resampled_means(values: Sequence[float], block: int, starts: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """The mean of every resample defined by ``starts``."""
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    prefix = np.concatenate([[0.0], np.cumsum(np.concatenate([x, x]))])
    last = n - (starts.shape[1] - 1) * block  # the final block is truncated so each resample has n observations
    full = prefix[starts[:, :-1] + block] - prefix[starts[:, :-1]]
    tail = prefix[starts[:, -1] + last] - prefix[starts[:, -1]]
    out: npt.NDArray[np.float64] = (full.sum(axis=1) + tail) / n
    return out


def interval(values: Sequence[float], block: int, *, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile interval of the mean, with the same quantile convention as research."""
    means = np.sort(resampled_means(values, block, block_starts(len(values), block, seed=seed)))
    tail = (1 - level) / 2
    return float(means[int(tail * len(means))]), float(means[int((1 - tail) * len(means)) - 1])


def standard_error(values: Sequence[float], block: int, *, seed: int = 0) -> float:
    """The bootstrap standard error of the mean: the standard deviation of the resampled means."""
    means = resampled_means(values, block, block_starts(len(values), block, seed=seed))
    return float(np.std(means, ddof=1))
