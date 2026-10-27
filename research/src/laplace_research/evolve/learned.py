"""The learned layer: a small network that corrects EV-HAR's bin probabilities.

EV-HAR gives a distribution for every question. The network sees a few dozen inputs a program can compute from its
context and adds a correction to EV-HAR's log-probabilities, so it starts from EV-HAR and only learns where EV-HAR is
wrong:

    logits = log EV-HAR(bins) + W2 tanh(W1 standardize(x) + b1) + b2

It is fitted per horizon on the binned log score (multiclass log loss) with a light L2 penalty, every hour of a fit
window, stopping at the best score on a later validation window. Weights are stored as 16-bit integers with one scale
per tensor, so the frozen program carries exactly the weights the research forecasts use.

``inputs`` computes the network's inputs for whole panels; ``INPUTS_PROGRAM`` computes the same numbers from one
context, inside a frozen program (tested to agree).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import optimize

from laplace_research.data.events import Calendar
from laplace_research.evolve.baselines import (
    Baseline,
    hourly_returns,
    implied_hourly_variance,
    overlay_features,
    overlay_names,
    trailing_mean,
)
from laplace_research.evolve.feed import FLOOR, HORIZONS, N_BINS, FloatArray, IntArray, edges

HOUR = 3600
DAY = 86_400
HIDDEN = 8
LONGEST = 719  # hours in a program's longest window of returns
L2 = 1e-4
MAX_ITERATIONS = 400
CHECK_EVERY = 10  # iterations between validation checks
SCALE_BITS = 15  # weights are stored as integers in [-2^15, 2^15)
OVERLAY = ("weekend", "releases", "recent-releases", "upcoming-releases", "us-open", "us-session")
NAMES = (
    "log_sd_over_edge", "rv1", "rv4", "rv24", "rv720", "iv", "ret1", "ret4", "ret24", "abs_ret1", "park1", "park24",
    "btc_rv1", "btc_ret1", "funding", "hour_sin", "hour_cos", "hour2_sin", "hour2_cos", "weekday_sin", "weekday_cos",
    *OVERLAY,
)  # fmt: skip


def final(p: FloatArray) -> FloatArray:
    """``feed.finalize``: normalize, then mix with the uniform distribution at weight ``FLOOR``."""
    out: FloatArray = (1 - FLOOR) * p / p.sum(axis=-1, keepdims=True) + FLOOR / N_BINS
    return out


@dataclass
class Panelwise:
    """Everything the inputs need, computed once for a panel (arrays are ``[asset, hour]``)."""

    means: dict[int, FloatArray]
    implied: FloatArray
    returns: dict[int, FloatArray]
    parkinson: dict[int, FloatArray]
    funding: FloatArray
    btc: int

    @classmethod
    def of(
        cls,
        close: FloatArray,
        high: FloatArray,
        low: FloatArray,
        senses: dict[str, FloatArray],
        assets: tuple[str, ...],
    ) -> Panelwise:
        r = hourly_returns(close)
        filled = np.where(np.isfinite(senses["rv"]), senses["rv"], r * r)
        # A program's window holds 720 candles, so 719 hourly returns: its longest mean covers 719 hours, and so does
        # this one, so research and the frozen program compute the same inputs.
        means = {w: trailing_mean(filled, w) for w in (1, 4, 24, 168, LONGEST)}
        implied = implied_hourly_variance(senses["dvol"], assets)
        implied = np.where(np.isfinite(implied), implied, means[LONGEST])
        cum = np.cumsum(r, axis=1)
        returns = {}
        for k in (1, 4, 24):
            out = np.zeros_like(r)
            out[:, k:] = cum[:, k:] - cum[:, :-k]
            returns[k] = out
        hl = np.log(high / low) ** 2 / (4 * math.log(2))
        parkinson = {w: trailing_mean(hl, w) for w in (1, 24)}
        funding = np.nan_to_num(senses["funding"]) * 1e4
        return cls(means, implied, returns, parkinson, funding, assets.index("BTC"))


def inputs(
    pw: Panelwise, calendar: Calendar, ts: IntArray, times: IntArray, sd: FloatArray, horizon: int
) -> FloatArray:
    """``[asset, sample, input]``: the network's inputs at panel hours ``times`` for one horizon; ``sd`` is EV-HAR's
    standard deviation ``[asset, sample]`` at those hours."""
    h = horizon
    at = ts[times] + HOUR
    names = overlay_names(h)
    overlay = overlay_features(calendar, at, h)
    picked = [overlay[:, names.index(k)] for k in OVERLAY]
    hour, weekday = (at // HOUR) % 24, ((at // DAY) + 3) % 7
    clock = [
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * hour / 12),
        np.cos(2 * np.pi * hour / 12),
        np.sin(2 * np.pi * weekday / 7),
        np.cos(2 * np.pi * weekday / 7),
    ]
    b = pw.btc
    rows = []
    for a in range(sd.shape[0]):
        w = np.maximum(pw.means[168][a, times], 1e-12)
        bw = np.maximum(pw.means[168][b, times], 1e-12)
        cols = [
            np.log(np.maximum(sd[a], 1e-12) / edges(h)[5]),
            *(np.log(np.maximum(pw.means[k][a, times], 1e-12) / w) for k in (1, 4, 24, LONGEST)),
            np.log(np.maximum(pw.implied[a, times], 1e-12) / w),
            pw.returns[1][a, times] / np.sqrt(w),
            pw.returns[4][a, times] / np.sqrt(4 * w),
            pw.returns[24][a, times] / np.sqrt(24 * w),
            np.abs(pw.returns[1][a, times]) / np.sqrt(w),
            np.log(np.maximum(pw.parkinson[1][a, times], 1e-12) / w),
            np.log(np.maximum(pw.parkinson[24][a, times], 1e-12) / w),
            np.log(np.maximum(pw.means[1][b, times], 1e-12) / bw),
            pw.returns[1][b, times] / np.sqrt(bw),
            pw.funding[a, times],
            *clock,
            *picked,
        ]
        rows.append(np.stack(cols, axis=-1))
    out: FloatArray = np.nan_to_num(np.stack(rows), nan=0.0, posinf=0.0, neginf=0.0)
    return out


def softmax(z: FloatArray) -> FloatArray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    out: FloatArray = e / e.sum(axis=-1, keepdims=True)
    return out


def _quantize(x: FloatArray) -> tuple[list[int], float]:
    """Integers and one scale such that ``ints * scale`` reproduces ``x`` to 2^-15 of its largest magnitude."""
    top = float(np.max(np.abs(x))) if x.size else 0.0
    scale = top / (2**SCALE_BITS - 1) if top > 0 else 1.0
    return [int(v) for v in np.round(x / scale).astype(np.int64).ravel()], scale


@dataclass
class Net:
    """One horizon's network, with its weights as stored (integers and a scale per tensor)."""

    mu: FloatArray
    sd: FloatArray
    w1: FloatArray  # [inputs, hidden]
    b1: FloatArray
    w2: FloatArray  # [hidden, bins]
    b2: FloatArray

    def logits(self, x: FloatArray) -> FloatArray:
        out: FloatArray = np.tanh(((x - self.mu) / self.sd) @ self.w1 + self.b1) @ self.w2 + self.b2
        return out

    def stored(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in ("mu", "sd", "w1", "b1", "w2", "b2"):
            ints, scale = _quantize(getattr(self, name))
            out[name] = {"shape": list(getattr(self, name).shape), "scale": scale, "ints": ints}
        return out

    @classmethod
    def from_stored(cls, stored: dict[str, Any]) -> Net:
        def tensor(name: str) -> FloatArray:
            t = stored[name]
            out: FloatArray = (np.asarray(t["ints"], dtype=np.float64) * t["scale"]).reshape(t["shape"])
            return out

        return cls(*(tensor(n) for n in ("mu", "sd", "w1", "b1", "w2", "b2")))


def fit_net(
    x: FloatArray,
    base: FloatArray,
    y: IntArray,
    xv: FloatArray,
    basev: FloatArray,
    yv: IntArray,
    *,
    hidden: int = HIDDEN,
    seed: int = 0,
) -> Net:
    """Fit on rows ``x`` (inputs), ``base`` (EV-HAR's final probabilities) and ``y`` (realized bins); keep the weights
    that score best on the validation rows. The returned net is already rounded to its stored form."""
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)  # an input that never varies carries nothing: pass it unscaled
    xs, xvs = (x - mu) / sd, (xv - mu) / sd
    lb, lbv = np.log(base), np.log(basev)
    k = x.shape[1]
    shapes = [(k, hidden), (hidden,), (hidden, N_BINS), (N_BINS,)]
    sizes = [int(np.prod(s)) for s in shapes]
    rng = np.random.default_rng(seed)
    theta0 = np.concatenate([rng.normal(0, 1 / math.sqrt(k), sizes[0]), np.zeros(sum(sizes[1:]))])

    def unpack(theta: FloatArray) -> list[FloatArray]:
        out, i = [], 0
        for s, n in zip(shapes, sizes, strict=True):
            out.append(theta[i : i + n].reshape(s))
            i += n
        return out

    def loss(theta: FloatArray, xs: FloatArray, lb: FloatArray, y: IntArray) -> tuple[float, FloatArray]:
        w1, b1, w2, b2 = unpack(theta)
        hid = np.tanh(xs @ w1 + b1)
        p = softmax(lb + hid @ w2 + b2)
        n = len(y)
        value = -float(np.mean(np.log(p[np.arange(n), y]))) + L2 * float(np.sum(w1**2) + np.sum(w2**2))
        g = p.copy()
        g[np.arange(n), y] -= 1
        g /= n
        dh = (g @ w2.T) * (1 - hid**2)
        grad = [xs.T @ dh + 2 * L2 * w1, dh.sum(axis=0), hid.T @ g + 2 * L2 * w2, g.sum(axis=0)]
        return value, np.concatenate([t.ravel() for t in grad])

    def score(theta: FloatArray) -> float:
        w1, b1, w2, b2 = unpack(theta)
        p = softmax(lbv + np.tanh(xvs @ w1 + b1) @ w2 + b2)
        return -float(np.mean(np.log(p[np.arange(len(yv)), yv])))

    best_value, best_theta = [score(theta0)], [theta0]
    count = [0]

    def check(theta: FloatArray) -> None:
        count[0] += 1
        if count[0] % CHECK_EVERY == 0:
            value = score(theta)
            if value < best_value[0]:
                best_value[0], best_theta[0] = value, theta.copy()

    optimize.minimize(
        loss,
        theta0,
        args=(xs, lb, y),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": MAX_ITERATIONS},
        callback=check,
    )
    w1, b1, w2, b2 = unpack(best_theta[0])
    net = Net.from_stored(Net(mu, sd, w1, b1, w2, b2).stored())
    if np.any(net.sd == 0):
        raise ValueError("an input's spread rounds to zero in 16 bits: rescale that input")
    return net


@dataclass
class Learned:
    """EV-HAR corrected by one network per horizon: ``probs(times)`` gives ``[asset, horizon, sample, bin]`` raw
    probabilities (finalize them once before scoring), exactly what the frozen program returns."""

    name: str
    ev: Baseline
    panelwise: Panelwise
    calendar: Calendar
    ts: IntArray
    nets: dict[int, Net]
    params: dict[str, Any] = field(default_factory=dict)

    def probs(self, times: IntArray) -> FloatArray:
        base = final(self.ev.probs(times))  # [asset, horizon, sample, bin]
        sd = self.ev.sd(times)
        out = np.empty_like(base)
        for j, h in enumerate(HORIZONS):
            x = inputs(self.panelwise, self.calendar, self.ts, times, sd[:, j], h)
            out[:, j] = softmax(np.log(base[:, j]) + self.nets[h].logits(x))
        return out


def ev_net(
    ev: Baseline,
    panelwise: Panelwise,
    calendar: Calendar,
    ts: IntArray,
    fit_hours: IntArray,
    valid_hours: IntArray,
    realized: Callable[[IntArray], IntArray],
) -> Learned:
    """Fit EV-HAR's correction on every hour of ``fit_hours``, keeping the weights best on ``valid_hours``."""
    nets: dict[int, Net] = {}
    rows = {}
    for name, hours in (("fit", fit_hours), ("valid", valid_hours)):
        rows[name] = (final(ev.probs(hours)), ev.sd(hours), realized(hours))
    for j, h in enumerate(HORIZONS):
        (bf, sf, yf), (bv, sv, yv) = rows["fit"], rows["valid"]
        x = inputs(panelwise, calendar, ts, fit_hours, sf[:, j], h).reshape(-1, len(NAMES))
        xv = inputs(panelwise, calendar, ts, valid_hours, sv[:, j], h).reshape(-1, len(NAMES))
        nets[h] = fit_net(
            x,
            bf[:, j].reshape(-1, N_BINS),
            yf[:, j].reshape(-1),
            xv,
            bv[:, j].reshape(-1, N_BINS),
            yv[:, j].reshape(-1),
        )
    params = ev.params | {"net": {str(h): nets[h].stored() for h in HORIZONS}, "inputs": list(NAMES), "hidden": HIDDEN}
    return Learned("ev-net", ev, panelwise, calendar, ts, nets, params)
