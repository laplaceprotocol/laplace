"""The baseline ladder every evolved program is measured against.

- ``naive``: a zero-mean Laplace distribution scaled by the last week's realized volatility (the seed program).
- ``ewma``: an exponentially weighted variance (RiskMetrics) with Student-t tails.
- ``garch``: GARCH(1,1) with Student-t innovations, fitted per asset by maximum likelihood, with exact multi-step
  variance forecasts.
- ``har``: a heterogeneous autoregressive model of future realized variance on 1-, 24-, 168- and 720-hour realized
  variance, fitted by least squares per asset and horizon, with Student-t tails.

Every parameter is fitted on the search window only. Tail shapes (``nu``) are chosen per horizon by the same binned log
score the programs are judged by. Each fitted baseline exposes its parameters (``params``, JSON-ready), so it can be
frozen and rerun elsewhere.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import optimize, signal, special

from laplace_research.evolve.feed import HORIZONS, N_BINS, FloatArray, IntArray, edges, finalize

NU_GRID = (2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0, 50.0)
LAMBDA_GRID = (0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998)
HAR_WINDOWS = (1, 24, 168, 720)


def hourly_returns(close: FloatArray) -> FloatArray:
    """``r[a, t] = log(close[a, t] / close[a, t - 1])``, with ``r[a, 0] = 0``."""
    r = np.zeros_like(close)
    r[:, 1:] = np.diff(np.log(close), axis=1)
    return r


def trailing_mean(x: FloatArray, window: int) -> FloatArray:
    """Mean of ``x[..., t - window + 1 : t + 1]`` (shorter at the start)."""
    c = np.cumsum(x, axis=-1)
    out = c.copy()
    out[..., window:] = c[..., window:] - c[..., :-window]
    counts = np.minimum(np.arange(1, x.shape[-1] + 1), window)
    result: FloatArray = out / counts
    return result


def t_bins(sd: FloatArray, nu: float, horizon: int) -> FloatArray:
    """Student-t bin probabilities for an array of standard deviations: shape ``sd.shape + (N_BINS,)``."""
    scale = np.maximum(sd, 1e-9) * math.sqrt((nu - 2) / nu)
    cdf = special.stdtr(nu, edges(horizon) / scale[..., None])
    zeros = np.zeros((*sd.shape, 1))
    out: FloatArray = np.diff(np.concatenate([zeros, cdf, zeros + 1], axis=-1), axis=-1)
    return out


def laplace_bins(sd: FloatArray, horizon: int) -> FloatArray:
    b = np.maximum(sd, 1e-9)[..., None] / math.sqrt(2)
    z = edges(horizon) / b
    cdf = np.where(z < 0, 0.5 * np.exp(np.minimum(z, 0)), 1 - 0.5 * np.exp(-np.maximum(z, 0)))
    zeros = np.zeros((*sd.shape, 1))
    out: FloatArray = np.diff(np.concatenate([zeros, cdf, zeros + 1], axis=-1), axis=-1)
    return out


def _mean_log_score(probs: FloatArray, realized: IntArray) -> float:
    p = finalize(probs)
    return float(np.mean(np.log(np.take_along_axis(p, realized[..., None], axis=-1)[..., 0])))


def _best_nu(sd: FloatArray, realized: IntArray, horizon: int) -> float:
    return max(NU_GRID, key=lambda nu: _mean_log_score(t_bins(sd, nu, horizon), realized))


@dataclass
class Baseline:
    """A fitted baseline: ``sd(times)`` gives ``[asset, horizon, sample]`` standard deviations; ``probs`` returns raw
    bin probabilities (finalize them once before scoring). ``params`` holds every other fitted value, per asset in panel
    order where it differs by asset."""

    name: str
    sd: Callable[[IntArray], FloatArray]
    nu: dict[int, float] = field(default_factory=dict)  # empty for the Laplace baseline
    params: dict[str, Any] = field(default_factory=dict)

    def probs(self, times: IntArray) -> FloatArray:
        sd = self.sd(times)
        out = np.empty((*sd.shape, N_BINS))
        for j, h in enumerate(HORIZONS):
            out[:, j] = laplace_bins(sd[:, j], h) if not self.nu else t_bins(sd[:, j], self.nu[h], h)
        return out


def naive(close: FloatArray) -> Baseline:
    week_var = trailing_mean(hourly_returns(close) ** 2, 168)

    def sd(times: IntArray) -> FloatArray:
        return np.sqrt(np.stack([week_var[:, times] * h for h in HORIZONS], axis=1))

    return Baseline("naive", sd)


def _fit_nu(sd: Callable[[IntArray], FloatArray], realized: IntArray, fit_times: IntArray) -> dict[int, float]:
    s = sd(fit_times)
    return {h: _best_nu(s[:, j], realized[:, j], h) for j, h in enumerate(HORIZONS)}


def ewma(close: FloatArray, realized_fit: IntArray, fit_times: IntArray) -> Baseline:
    r2 = hourly_returns(close) ** 2

    def variance(lam: float) -> FloatArray:
        init = r2[:, 1:169].mean(axis=1, keepdims=True)
        v, _ = signal.lfilter([1 - lam], [1, -lam], r2, axis=1, zi=lam * init)
        out: FloatArray = v
        return out

    def make_sd(v: FloatArray) -> Callable[[IntArray], FloatArray]:
        return lambda times: np.sqrt(np.stack([v[:, times] * h for h in HORIZONS], axis=1))

    best: tuple[float, float, dict[int, float]] | None = None
    for lam in LAMBDA_GRID:
        sd = make_sd(variance(lam))
        nus = _fit_nu(sd, realized_fit, fit_times)
        score = sum(
            _mean_log_score(t_bins(sd(fit_times)[:, j], nus[h], h), realized_fit[:, j]) for j, h in enumerate(HORIZONS)
        )
        if best is None or score > best[0]:
            best = (score, lam, nus)
    assert best is not None
    _, lam, nus = best
    return Baseline(f"ewma(lambda={lam})", make_sd(variance(lam)), nus, {"lambda": lam})


def _garch_variance(r: FloatArray, omega: float, alpha: float, beta: float) -> FloatArray:
    """Conditional variance of ``r[t + 1]`` given returns up to ``t``: ``v[t] = omega + alpha r[t]^2 + beta v[t-1]``."""
    init = float(np.mean(r[: min(len(r), 720)] ** 2))
    steady = omega / (1 - beta)
    # y[t] = alpha r[t]^2 + beta y[t-1] with y = v - steady reproduces the recursion exactly, from v[-1] = init.
    y, _ = signal.lfilter([alpha], [1, -beta], r**2, zi=[beta * (init - steady)])
    out: FloatArray = y + steady
    return out


def _t_log_density(x: FloatArray, variance: FloatArray, nu: float) -> FloatArray:
    """Log density of a Student-t distribution with ``nu > 2`` degrees of freedom, rescaled to ``variance``."""
    out: FloatArray = (
        special.gammaln((nu + 1) / 2)
        - special.gammaln(nu / 2)
        - 0.5 * np.log(np.pi * (nu - 2) * variance)
        - (nu + 1) / 2 * np.log1p(x**2 / ((nu - 2) * variance))
    )
    return out


def _garch_fit(r: FloatArray) -> tuple[float, float, float]:
    var = float(np.var(r))

    def nll(params: FloatArray) -> float:
        a, b, nu = params
        if a < 0 or b < 0 or a + b >= 0.9999 or nu <= 2.05:
            return 1e12
        omega = var * (1 - a - b)
        v = np.maximum(_garch_variance(r[:-1], omega, a, b), 1e-14)
        return float(-np.sum(_t_log_density(r[1:], v, nu)))

    best = optimize.minimize(nll, x0=np.array([0.05, 0.9, 5.0]), method="Nelder-Mead", options={"maxiter": 600})
    a, b, _ = best.x
    return var * (1 - a - b), float(a), float(b)


def garch(close: FloatArray, realized_fit: IntArray, fit_times: IntArray, fit_range: slice) -> Baseline:
    r = hourly_returns(close)
    one_step = np.empty_like(r)
    persistence = np.empty(len(r))
    long_run = np.empty(len(r))
    fitted = []
    for a in range(r.shape[0]):
        omega, alpha, beta = _garch_fit(r[a, fit_range])
        fitted.append({"omega": omega, "alpha": alpha, "beta": beta})
        one_step[a] = _garch_variance(r[a], omega, alpha, beta)
        persistence[a] = alpha + beta
        long_run[a] = omega / (1 - alpha - beta)

    def sd(times: IntArray) -> FloatArray:
        out = np.empty((r.shape[0], len(HORIZONS), len(times)))
        for j, h in enumerate(HORIZONS):
            # Sum of k-step variance forecasts: V_L + p^(k-1) (v1 - V_L), k = 1..h.
            geometric = np.array([np.sum(p ** np.arange(h)) for p in persistence])
            v1 = one_step[:, times]
            out[:, j] = np.sqrt(
                np.maximum(h * long_run[:, None] + geometric[:, None] * (v1 - long_run[:, None]), 1e-12)
            )
        return out

    return Baseline("garch(1,1)-t", sd, _fit_nu(sd, realized_fit, fit_times), {"per_asset": fitted})


def har(close: FloatArray, realized_fit: IntArray, fit_times: IntArray) -> Baseline:
    r2 = hourly_returns(close) ** 2
    features = np.stack([trailing_mean(r2, w) for w in HAR_WINDOWS], axis=-1)  # [asset, t, k]
    future = np.cumsum(r2, axis=1)
    coefs: dict[tuple[int, int], FloatArray] = {}
    for a in range(r2.shape[0]):
        for h in HORIZONS:
            t = fit_times
            y = future[a, t + h] - future[a, t]
            x = np.column_stack([np.ones(len(t)), features[a, t] * h])
            coefs[a, h], *_ = np.linalg.lstsq(x, y, rcond=None)

    def sd(times: IntArray) -> FloatArray:
        out = np.empty((r2.shape[0], len(HORIZONS), len(times)))
        for a in range(r2.shape[0]):
            for j, h in enumerate(HORIZONS):
                x = np.column_stack([np.ones(len(times)), features[a, times] * h])
                floor_var = 0.1 * features[a, times, 2] * h  # never below a tenth of the weekly variance
                out[a, j] = np.sqrt(np.maximum(x @ coefs[a, h], floor_var))
        return out

    params = {
        "windows": list(HAR_WINDOWS),
        # [intercept, one coefficient per window] for each asset and horizon; the model is fitted on horizon-scaled
        # trailing variances, so the variance forecast is ``c0 + sum(c_k * mean_k * h)``.
        "per_asset": [{str(h): [float(c) for c in coefs[a, h]] for h in HORIZONS} for a in range(r2.shape[0])],
    }
    return Baseline("har-rv", sd, _fit_nu(sd, realized_fit, fit_times), params)


HOURS_PER_YEAR = 365 * 24


def implied_hourly_variance(dvol: FloatArray, assets: tuple[str, ...]) -> FloatArray:
    """Hourly variance implied by DVOL (annualized percent): the asset's own index, else BTC's as the market's."""
    market = dvol[assets.index("BTC")]
    own_or_market = np.where(np.isfinite(dvol), dvol, market[None, :])
    out: FloatArray = (own_or_market / 100) ** 2 / HOURS_PER_YEAR
    return out


def _regression_baseline(
    name: str,
    close: FloatArray,
    features: FloatArray,
    floor: FloatArray,
    realized_fit: IntArray,
    fit_times: IntArray,
    params: dict[str, Any],
    forecast_features: FloatArray | None = None,
) -> Baseline:
    """A baseline whose h-hour variance is ``c0 + h * sum(c_k * features_k)``, fitted by least squares on the realized
    squared hourly returns over the horizon (the question's own variance), floored at ``floor * h``. Fit rows with a
    missing feature are skipped; forecasts use ``forecast_features`` (the same features with gaps filled) if given."""
    r2 = hourly_returns(close) ** 2
    future = np.cumsum(r2, axis=1)
    coefs: dict[tuple[int, int], FloatArray] = {}
    for a in range(r2.shape[0]):
        for h in HORIZONS:
            t = fit_times[np.isfinite(features[a, fit_times]).all(axis=1)]
            y = future[a, t + h] - future[a, t]
            x = np.column_stack([np.ones(len(t)), features[a, t] * h])
            coefs[a, h], *_ = np.linalg.lstsq(x, y, rcond=None)

    at_forecast = features if forecast_features is None else forecast_features

    def sd(times: IntArray) -> FloatArray:
        out = np.empty((r2.shape[0], len(HORIZONS), len(times)))
        for a in range(r2.shape[0]):
            for j, h in enumerate(HORIZONS):
                x = np.column_stack([np.ones(len(times)), at_forecast[a, times] * h])
                variance = np.nan_to_num(x @ coefs[a, h], nan=0.0)
                out[a, j] = np.sqrt(np.maximum(variance, floor[a, times] * h))
        return out

    params = params | {
        "per_asset": [{str(h): [float(c) for c in coefs[a, h]] for h in HORIZONS} for a in range(r2.shape[0])]
    }
    return Baseline(name, sd, _fit_nu(sd, realized_fit, fit_times), params)


def rv_har(close: FloatArray, rv: FloatArray, realized_fit: IntArray, fit_times: IntArray) -> Baseline:
    """HAR on intraday realized variance: the 1-, 24-, 168- and 720-hour means of 5-minute realized variance (hours
    without it use the squared hourly return), floored at a tenth of the weekly mean."""
    r2 = hourly_returns(close) ** 2
    filled = np.where(np.isfinite(rv), rv, r2)
    features = np.stack([trailing_mean(filled, w) for w in HAR_WINDOWS], axis=-1)
    params = {"windows": list(HAR_WINDOWS), "features": "trailing means of 5-minute realized variance"}
    return _regression_baseline("rv-har", close, features, 0.1 * features[..., 2], realized_fit, fit_times, params)


def iv_har(
    close: FloatArray,
    rv: FloatArray,
    dvol: FloatArray,
    assets: tuple[str, ...],
    realized_fit: IntArray,
    fit_times: IntArray,
) -> Baseline:
    """RV-HAR plus the hourly variance implied by DVOL (own index for BTC and ETH, BTC's for the others)."""
    r2 = hourly_returns(close) ** 2
    filled = np.where(np.isfinite(rv), rv, r2)
    windows = np.stack([trailing_mean(filled, w) for w in HAR_WINDOWS], axis=-1)
    implied = implied_hourly_variance(dvol, assets)
    features = np.concatenate([windows, implied[..., None]], axis=-1)
    # Where no index is known at forecast time, the month's realized variance stands in for the implied one.
    filled = np.concatenate([windows, np.where(np.isfinite(implied), implied, windows[..., 3])[..., None]], axis=-1)
    params = {
        "windows": list(HAR_WINDOWS),
        "features": "trailing means of 5-minute realized variance, then DVOL-implied hourly variance",
    }
    return _regression_baseline(
        "iv-har", close, features, 0.1 * windows[..., 2], realized_fit, fit_times, params, forecast_features=filled
    )


def ladder(
    close: FloatArray,
    realized_fit: IntArray,
    fit_times: IntArray,
    fit_range: slice,
    *,
    senses: dict[str, FloatArray] | None = None,
    assets: tuple[str, ...] | None = None,
) -> list[Baseline]:
    """The classical ladder; with senses, also RV-HAR (needs ``rv``) and IV-HAR (also needs ``dvol`` and ``assets``)."""
    rungs = [
        naive(close),
        ewma(close, realized_fit, fit_times),
        garch(close, realized_fit, fit_times, fit_range),
        har(close, realized_fit, fit_times),
    ]
    senses = senses or {}
    if "rv" in senses:
        rungs.append(rv_har(close, senses["rv"], realized_fit, fit_times))
        if "dvol" in senses and assets is not None:
            rungs.append(iv_har(close, senses["rv"], senses["dvol"], assets, realized_fit, fit_times))
    return rungs
