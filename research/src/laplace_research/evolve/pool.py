"""Linear pools of final forecasts, fitted by EM with a convergence certificate.

For members' probabilities ``p[m, s]`` of the realized outcomes, the pool maximizes ``f(w) = mean_s log(w @ p[:, s])``
over the simplex. With ``g_m = mean_s p[m, s] / (w @ p[:, s])``, concavity gives the certificate
``f(w*) - f(w) <= max_m g_m - 1``, whatever path the fit took to ``w``, so the fit stops only when that gap is below a
tolerance far smaller than the contributions being measured.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from laplace_research.evolve.feed import FloatArray, IntArray

DEFAULT_TOLERANCE = 1e-8
ETA_GROWTH = 1.2  # over-relaxation grows by this factor after each step that raised the objective
ETA_MAX = 1e6


class PoolDidNotConverge(RuntimeError):
    pass


@dataclass(frozen=True)
class PoolFit:
    weights: FloatArray
    gap: float  # certified upper bound on the remaining objective shortfall, nats per sample
    iterations: int


def fit_pool(p: FloatArray, *, tolerance: float = DEFAULT_TOLERANCE, max_iterations: int = 200_000) -> PoolFit:
    """EM for mixture weights, adaptively over-relaxed. ``p``: members x samples, strictly positive (final forecasts
    are).

    EM multiplies each weight by ``g_m``; plain EM crawls when members are near-copies of each other. The
    over-relaxed step multiplies it by ``g_m ** eta``: ``eta`` grows while steps raise the objective, and drops back
    to 1, plain EM, which never lowers it, when a step would not (Salakhutdinov and Roweis, 2003). Weights are kept as
    logarithms, so none underflows to an exact zero it could never leave.
    """
    log_w = np.full(p.shape[0], -np.log(p.shape[0]))
    w = np.exp(log_w)
    mixed = w @ p
    objective = float(np.mean(np.log(mixed)))
    eta = 1.0
    for i in range(max_iterations):
        g = np.mean(p / mixed, axis=1)
        gap = float(g.max() - 1.0)
        if gap <= tolerance:
            return PoolFit(w, gap, i)
        log_g = np.log(g)
        while True:
            step = log_w + eta * log_g
            step -= np.logaddexp.reduce(step)
            step_w = np.exp(step)
            step_mixed = step_w @ p
            step_objective = float(np.mean(np.log(step_mixed)))
            if step_objective >= objective or eta == 1.0:
                break
            eta = 1.0
        log_w, w, mixed, objective = step, step_w, step_mixed, step_objective
        eta = min(eta * ETA_GROWTH, ETA_MAX)
    raise PoolDidNotConverge(f"gap {gap:.2e} after {max_iterations} iterations")


def realized_probs(final: FloatArray, realized: IntArray) -> FloatArray:
    """Probability that final forecasts gave the realized bin, flattened over asset, horizon and sample."""
    out: FloatArray = np.take_along_axis(final, realized[..., None], axis=-1)[..., 0].reshape(-1)
    return out


def mix(weights: FloatArray, finals: Sequence[FloatArray]) -> FloatArray:
    """The pooled final forecast. A mixture of final forecasts is final: it is never smoothed again."""
    out: FloatArray = np.tensordot(weights, np.stack(finals), axes=1)
    return out


def renormalized_contributions(weights: FloatArray, p: FloatArray) -> FloatArray:
    """Pool score minus the score with one member removed and the other weights renormalized (not refitted).

    A diagnostic of how much the fitted pool leans on each member. It is not a payout rule: it can credit members
    whose errors merely cancel, and copies can keep positive credit.
    """
    full = float(np.mean(np.log(weights @ p)))
    out = np.empty(len(weights))
    for k in range(len(weights)):
        keep = np.arange(len(weights)) != k
        w = weights[keep]
        out[k] = full - float(np.mean(np.log((w / w.sum()) @ p[keep]))) if w.sum() > 0 else np.nan
    return out
