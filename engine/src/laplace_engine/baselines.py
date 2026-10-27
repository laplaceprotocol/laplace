"""The baseline ladder as forecasting programs, with parameters frozen from a research fit.

Baselines run exactly like evolved programs: the same sandbox, the same past-only window, and their source is public.
Each program reproduces its research implementation (``laplace_research.evolve.baselines``) on the 720-hour window a
program sees. Two details differ, both far below a basis point: recursions start from the window's own mean squared
return rather than the full history's, and HAR's 720-hour mean covers the window's 719 returns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from laplace_research.evolve.baselines import Baseline
from laplace_research.evolve.islands import SEED_PROGRAM

NAIVE = SEED_PROGRAM

MONKEY = '''\
def forecast(ctx):
    """The monkey: equal odds on every bin. Any forecaster worth its name beats it by a wide margin."""
    return [1.0] * 10
'''

EWMA = '''\
import math
import numpy as np

# LAMBDA: the decay. NU: Student-t degrees of freedom per horizon, chosen by binned log score.
# Both fitted on {origin}.
LAMBDA = {lam!r}
NU = {nu}


def forecast(ctx):
    """EWMA (RiskMetrics) variance of hourly returns, with Student-t tails."""
    r = np.diff(np.log(ctx.close))
    r2 = r * r
    var = float(np.mean(r2[:168]))
    for x in r2:
        var = LAMBDA * var + (1 - LAMBDA) * float(x)
    return ctx.lib.student_t_probs(math.sqrt(var * ctx.horizon), NU[ctx.horizon], ctx.edges)
'''

GARCH = '''\
import math
import numpy as np

# Per asset: omega, alpha, beta, fitted by maximum likelihood with Student-t innovations.
# NU: Student-t degrees of freedom per horizon, chosen by binned log score.
# Both fitted on {origin}.
PARAMS = {{
{params}
}}
NU = {nu}


def forecast(ctx):
    """GARCH(1,1): the variance recursion over the window, then the exact variance summed over the horizon."""
    omega, alpha, beta = PARAMS[ctx.asset]
    r = np.diff(np.log(ctx.close))
    var = float(np.mean(r * r))
    for x in r:
        var = omega + alpha * float(x) ** 2 + beta * var
    persistence = alpha + beta
    long_run = omega / (1 - persistence)
    total = ctx.horizon * long_run + (var - long_run) * sum(persistence**k for k in range(ctx.horizon))
    return ctx.lib.student_t_probs(math.sqrt(max(total, 1e-12)), NU[ctx.horizon], ctx.edges)
'''

HAR = '''\
import math
import numpy as np

WINDOWS = {windows!r}
# Per asset and horizon: intercept, then one coefficient per window, fitted by least squares.
# NU: Student-t degrees of freedom per horizon, chosen by binned log score.
# Both fitted on {origin}.
COEFS = {{
{coefs}
}}
NU = {nu}


def forecast(ctx):
    """HAR-RV: the horizon's variance from 1-, 24-, 168- and 720-hour realized variance, with Student-t tails."""
    r = np.diff(np.log(ctx.close))
    r2 = r * r
    means = [float(np.mean(r2[-w:])) for w in WINDOWS]
    c = COEFS[ctx.asset][ctx.horizon]
    var = c[0] + sum(k * m for k, m in zip(c[1:], means)) * ctx.horizon
    floor = 0.1 * means[2] * ctx.horizon
    return ctx.lib.student_t_probs(math.sqrt(max(var, floor)), NU[ctx.horizon], ctx.edges)
'''


SENSES_RV = '''\
def _filled(ctx):
    """Hourly realized variance from 5-minute returns over the window, the squared hourly return where it is unknown."""
    r = np.diff(np.log(ctx.close))
    rv = ctx.senses.get("rv")
    if rv is None:
        return r * r
    rv = rv[1:]
    return np.where(np.isfinite(rv), rv, r * r)
'''

RV_HAR = '''\
import math
import numpy as np

WINDOWS = {windows!r}
# Per asset and horizon: intercept, then one coefficient per window of 5-minute realized variance, fitted by least
# squares. NU: Student-t degrees of freedom per horizon, chosen by binned log score.
# Both fitted on {origin}.
COEFS = {{
{coefs}
}}
NU = {nu}


{senses_rv}

def forecast(ctx):
    """RV-HAR: the horizon's variance from 1-, 24-, 168- and 720-hour means of 5-minute realized variance."""
    filled = _filled(ctx)
    means = [float(np.mean(filled[-w:])) for w in WINDOWS]
    c = COEFS[ctx.asset][ctx.horizon]
    var = c[0] + sum(k * m for k, m in zip(c[1:], means)) * ctx.horizon
    floor = 0.1 * means[2] * ctx.horizon
    return ctx.lib.student_t_probs(math.sqrt(max(var, floor)), NU[ctx.horizon], ctx.edges)
'''

IV_HAR = '''\
import math
import numpy as np

WINDOWS = {windows!r}
HOURS_PER_YEAR = 8760
# Per asset and horizon: intercept, one coefficient per window of 5-minute realized variance, then one for the hourly
# variance implied by DVOL, fitted by least squares. NU: Student-t degrees of freedom per horizon, chosen by binned
# log score. Both fitted on {origin}.
COEFS = {{
{coefs}
}}
NU = {nu}


{senses_rv}

def _implied(ctx):
    """The hourly variance implied by the asset's DVOL index at the forecast hour, else by BTC's (NaN if neither)."""
    value = math.nan
    own = ctx.senses.get("dvol")
    if own is not None:
        value = float(own[-1])
    if not math.isfinite(value):
        market = ctx.peer_senses.get("BTC", {{}}).get("dvol")
        value = float(market[-1]) if market is not None else math.nan
    return (value / 100) ** 2 / HOURS_PER_YEAR


def forecast(ctx):
    """IV-HAR: RV-HAR plus the variance implied by the options market; the month's realized variance stands in when
    no index is known."""
    filled = _filled(ctx)
    means = [float(np.mean(filled[-w:])) for w in WINDOWS]
    implied = _implied(ctx)
    if not math.isfinite(implied):
        implied = means[3]
    c = COEFS[ctx.asset][ctx.horizon]
    var = c[0] + (sum(k * m for k, m in zip(c[1:5], means)) + c[5] * implied) * ctx.horizon
    floor = 0.1 * means[2] * ctx.horizon
    return ctx.lib.student_t_probs(math.sqrt(max(var, floor)), NU[ctx.horizon], ctx.edges)
'''


EV_HAR = '''\
import math
import numpy as np

WINDOWS = {windows!r}
HOURS_PER_YEAR = 8760
RELEASES = ("fomc", "cpi", "nfp")
# Per asset and horizon: IV-HAR's intercept, one coefficient per window of 5-minute realized variance and one for the
# hourly variance implied by DVOL, fitted by least squares. BETA: the overlay's coefficients per horizon, in the order
# _inputs returns them, fitted by binned log score. NU: Student-t degrees of freedom per horizon, chosen by binned log
# score. All fitted on {origin}.
COEFS = {{
{coefs}
}}
BETA = {beta}
NU = {nu}


{senses_rv}

def _implied(ctx):
    """The hourly variance implied by the asset's DVOL index at the forecast hour, else by BTC's (NaN if neither)."""
    value = math.nan
    own = ctx.senses.get("dvol")
    if own is not None:
        value = float(own[-1])
    if not math.isfinite(value):
        market = ctx.peer_senses.get("BTC", {{}}).get("dvol")
        value = float(market[-1]) if market is not None else math.nan
    return (value / 100) ** 2 / HOURS_PER_YEAR


def _session_hours(events, horizon):
    """Hours of NYSE regular session inside the forecast's window."""
    total, opened = 0.0, None
    for hours, kind in events:
        if kind == "us-open":
            opened = hours
        elif kind == "us-close" and opened is not None:
            total += max(0.0, min(hours, horizon) - max(opened, 0.0))
            opened = None
    return total


def _inputs(ctx):
    """The overlay's inputs: the window's weekend share, the UTC hour (1- and 4-hour horizons), then the scheduled
    releases inside, just before and just after the window, NYSE opens inside it, and its share of the session."""
    h = ctx.horizon
    x = [sum(1 for k in range(h) if (ctx.weekday + (ctx.hour + k) // 24) % 7 >= 5) / h]
    if h <= 4:
        x += [(1.0 if ctx.hour == j else 0.0) - 1 / 24 for j in range(24)]
    releases = [t for t, kind in ctx.events if kind in RELEASES]
    x.append(sum(1 for t in releases if 0 <= t < h))
    x.append(sum(1 for t in releases if -3 <= t < 0))
    x.append(sum(1 for t in releases if h <= t < h + 2))
    x.append(sum(1 for t, kind in ctx.events if kind == "us-open" and 0 <= t < h))
    x.append(_session_hours(ctx.events, h) / h)
    return x


def forecast(ctx):
    """EV-HAR: IV-HAR's variance, scaled for the US session, weekends and scheduled releases."""
    filled = _filled(ctx)
    means = [float(np.mean(filled[-w:])) for w in WINDOWS]
    implied = _implied(ctx)
    if not math.isfinite(implied):
        implied = means[3]
    c = COEFS[ctx.asset][ctx.horizon]
    var = c[0] + (sum(k * m for k, m in zip(c[1:5], means)) + c[5] * implied) * ctx.horizon
    floor = 0.1 * means[2] * ctx.horizon
    scale = math.exp(0.5 * sum(b * x for b, x in zip(BETA[ctx.horizon], _inputs(ctx))))
    return ctx.lib.student_t_probs(math.sqrt(max(var, floor)) * scale, NU[ctx.horizon], ctx.edges)
'''


EV_NET = '''\
import math
import numpy as np

WINDOWS = {windows!r}
HOURS_PER_YEAR = 8760
RELEASES = ("fomc", "cpi", "nfp")
FLOOR = 0.001  # final forecasts mix in the uniform distribution at this weight
# EV-HAR, exactly as in its own program: per asset and horizon its intercept, one coefficient per window of 5-minute
# realized variance and one for the DVOL-implied variance (COEFS); its overlay's coefficients (BETA); its tails (NU).
# Then one small network per horizon that corrects EV-HAR's log-probabilities. NET holds each network's tensors, in the
# order of SHAPES, as 16-bit two's-complement integers in hex, each tensor scaled by its own factor. All fitted on
# {origin}.
COEFS = {{
{coefs}
}}
BETA = {beta}
NU = {nu}
SHAPES = {shapes}
NET = {{
{net}
}}


def _filled_of(close, rv):
    """Hourly realized variance from 5-minute returns over a window, the squared hourly return where it is unknown."""
    r = np.diff(np.log(close))
    if rv is None:
        return r * r
    rv = rv[1:]
    return np.where(np.isfinite(rv), rv, r * r)


def _filled(ctx):
    return _filled_of(ctx.close, ctx.senses.get("rv"))


def _implied(ctx):
    """The hourly variance implied by the asset's DVOL index at the forecast hour, else by BTC's (NaN if neither)."""
    value = math.nan
    own = ctx.senses.get("dvol")
    if own is not None:
        value = float(own[-1])
    if not math.isfinite(value):
        market = ctx.peer_senses.get("BTC", {{}}).get("dvol")
        value = float(market[-1]) if market is not None else math.nan
    return (value / 100) ** 2 / HOURS_PER_YEAR


def _session_hours(events, horizon):
    """Hours of NYSE regular session inside the forecast's window."""
    total, opened = 0.0, None
    for hours, kind in events:
        if kind == "us-open":
            opened = hours
        elif kind == "us-close" and opened is not None:
            total += max(0.0, min(hours, horizon) - max(opened, 0.0))
            opened = None
    return total


def _inputs(ctx):
    """EV-HAR's overlay inputs: the window's weekend share, the UTC hour (1- and 4-hour horizons), then the scheduled
    releases inside, just before and just after the window, NYSE opens inside it, and its share of the session."""
    h = ctx.horizon
    x = [sum(1 for k in range(h) if (ctx.weekday + (ctx.hour + k) // 24) % 7 >= 5) / h]
    if h <= 4:
        x += [(1.0 if ctx.hour == j else 0.0) - 1 / 24 for j in range(24)]
    releases = [t for t, kind in ctx.events if kind in RELEASES]
    x.append(sum(1 for t in releases if 0 <= t < h))
    x.append(sum(1 for t in releases if -3 <= t < 0))
    x.append(sum(1 for t in releases if h <= t < h + 2))
    x.append(sum(1 for t, kind in ctx.events if kind == "us-open" and 0 <= t < h))
    x.append(_session_hours(ctx.events, h) / h)
    return x


def _tensors(h):
    """Horizon h's network: its tensors, decoded from 16-bit integers and scaled."""
    text, scales = NET[h]
    ints = [int(text[i : i + 4], 16) for i in range(0, len(text), 4)]
    ints = [v - 65536 if v >= 32768 else v for v in ints]
    out, i = [], 0
    for shape, scale in zip(SHAPES, scales):
        n = 1
        for d in shape:
            n *= d
        out.append(np.array(ints[i : i + n], dtype=np.float64).reshape(shape) * scale)
        i += n
    return out


TENSORS = {{h: _tensors(h) for h in NET}}


def _net_inputs(ctx, sd):
    """The network's inputs (laplace_research.evolve.learned.NAMES, in that order) from the window and EV-HAR's sd."""
    filled = _filled(ctx)
    week = max(float(np.mean(filled[-168:])), 1e-12)
    means = {{w: max(float(np.mean(filled[-w:])), 1e-12) for w in (1, 4, 24, 720)}}
    implied = _implied(ctx)
    if not math.isfinite(implied):
        implied = float(np.mean(filled[-720:]))
    close = ctx.close
    ret = {{k: math.log(close[-1] / close[-1 - k]) for k in (1, 4, 24)}}
    hl = np.log(ctx.high / ctx.low) ** 2 / (4 * math.log(2))
    btc = ctx.peers["BTC"]
    btc_filled = _filled_of(btc.close, ctx.peer_senses.get("BTC", {{}}).get("rv"))
    btc_week = max(float(np.mean(btc_filled[-168:])), 1e-12)
    funding = ctx.senses.get("funding")
    funding = float(funding[-1]) * 1e4 if funding is not None else 0.0
    overlay = _inputs(ctx)
    x = [
        math.log(max(sd, 1e-12) / float(ctx.edges[5])),
        *(math.log(means[k] / week) for k in (1, 4, 24, 720)),
        math.log(max(implied, 1e-12) / week),
        ret[1] / math.sqrt(week),
        ret[4] / math.sqrt(4 * week),
        ret[24] / math.sqrt(24 * week),
        abs(ret[1]) / math.sqrt(week),
        math.log(max(float(hl[-1]), 1e-12) / week),
        math.log(max(float(np.mean(hl[-24:])), 1e-12) / week),
        math.log(max(float(btc_filled[-1]), 1e-12) / btc_week),
        math.log(btc.close[-1] / btc.close[-2]) / math.sqrt(btc_week),
        funding,
        math.sin(2 * math.pi * ctx.hour / 24),
        math.cos(2 * math.pi * ctx.hour / 24),
        math.sin(2 * math.pi * ctx.hour / 12),
        math.cos(2 * math.pi * ctx.hour / 12),
        math.sin(2 * math.pi * ctx.weekday / 7),
        math.cos(2 * math.pi * ctx.weekday / 7),
        overlay[0],
        *overlay[-5:],
    ]
    return [v if math.isfinite(v) else 0.0 for v in x]


def forecast(ctx):
    """EV-NET: EV-HAR's distribution, corrected by a small network trained on the binned log score."""
    filled = _filled(ctx)
    means = [float(np.mean(filled[-w:])) for w in WINDOWS]
    implied = _implied(ctx)
    if not math.isfinite(implied):
        implied = means[3]
    c = COEFS[ctx.asset][ctx.horizon]
    var = c[0] + (sum(k * m for k, m in zip(c[1:5], means)) + c[5] * implied) * ctx.horizon
    floor = 0.1 * means[2] * ctx.horizon
    scale = math.exp(0.5 * sum(b * x for b, x in zip(BETA[ctx.horizon], _inputs(ctx))))
    sd = math.sqrt(max(var, floor)) * scale
    base = np.asarray(ctx.lib.student_t_probs(sd, NU[ctx.horizon], ctx.edges), dtype=np.float64)
    base = (1 - FLOOR) * base / base.sum() + FLOOR / len(base)
    mu, spread, w1, b1, w2, b2 = TENSORS[ctx.horizon]
    x = (np.array(_net_inputs(ctx, sd)) - mu) / spread
    logits = np.log(base) + np.tanh(x @ w1 + b1) @ w2 + b2
    p = np.exp(logits - logits.max())
    return p / p.sum()
'''


def _hex16(ints: Sequence[int]) -> str:
    return "".join(f"{v & 0xFFFF:04x}" for v in ints)


NET_TENSORS = ("mu", "sd", "w1", "b1", "w2", "b2")


def learned_program(ev: Baseline, net: Mapping[str, Mapping[str, Any]], assets: Sequence[str], origin: str) -> str:
    """Source code for EV-NET from a fitted EV-HAR and its stored networks (``learned.Net.stored()`` per horizon)."""
    coefs = "\n".join(
        f"    {asset!r}: {{\n"
        + "".join(
            f"        {h}: ({', '.join(repr(float(c)) for c in by_h[str(h)])}),\n" for h in sorted(map(int, by_h))
        )
        + "    },"
        for asset, by_h in zip(assets, ev.params["per_asset"], strict=True)
    )
    beta = (
        "{\n"
        + "".join(
            f"    {h}: ({', '.join(repr(float(v)) for v in ev.params['overlay'][str(h)]['beta'])}),\n"
            for h in sorted(map(int, ev.params["overlay"]))
        )
        + "}"
    )
    first = net[sorted(net, key=int)[0]]
    shapes = tuple(tuple(first[t]["shape"]) for t in NET_TENSORS)
    rows = []
    for h in sorted(net, key=int):
        ints = [v for t in NET_TENSORS for v in net[h][t]["ints"]]
        scales = ", ".join(repr(float(net[h][t]["scale"])) for t in NET_TENSORS)
        rows.append(f'    {int(h)}: ("{_hex16(ints)}", ({scales})),')
    return EV_NET.format(
        windows=tuple(ev.params["windows"]),
        coefs=coefs,
        beta=beta,
        nu=_nu(ev.nu),
        shapes=repr(shapes),
        net="\n".join(rows),
        origin=origin,
    )


def _nu(nu: Mapping[int, float]) -> str:
    return "{" + ", ".join(f"{h}: {float(v)!r}" for h, v in sorted(nu.items())) + "}"


def ladder_programs(ladder: Sequence[Baseline], assets: Sequence[str], origin: str) -> dict[str, str]:
    """Source code for naive, EWMA, GARCH and HAR (and RV-HAR, IV-HAR and EV-HAR when the ladder has them) from a
    fitted research ladder, keyed by program name."""
    rungs = {b.name.split("(")[0]: b for b in ladder}
    ewma, garch, har = rungs["ewma"], rungs["garch"], rungs["har-rv"]

    def coefficients(rung: Baseline) -> str:
        return "\n".join(
            f"    {asset!r}: {{\n"
            + "".join(
                f"        {h}: ({', '.join(repr(float(c)) for c in by_h[str(h)])}),\n" for h in sorted(map(int, by_h))
            )
            + "    },"
            for asset, by_h in zip(assets, rung.params["per_asset"], strict=True)
        )

    garch_params = "\n".join(
        f"    {asset!r}: ({float(p['omega'])!r}, {float(p['alpha'])!r}, {float(p['beta'])!r}),"
        for asset, p in zip(assets, garch.params["per_asset"], strict=True)
    )
    har_coefs = "\n".join(
        f"    {asset!r}: {{\n"
        + "".join(
            f"        {h}: ({', '.join(repr(float(c)) for c in by_h[str(h)])}),\n" for h in sorted(map(int, by_h))
        )
        + "    },"
        for asset, by_h in zip(assets, har.params["per_asset"], strict=True)
    )
    programs = {
        "naive": NAIVE,
        "ewma": EWMA.format(lam=float(ewma.params["lambda"]), nu=_nu(ewma.nu), origin=origin),
        "garch": GARCH.format(params=garch_params, nu=_nu(garch.nu), origin=origin),
        "har": HAR.format(windows=tuple(har.params["windows"]), coefs=har_coefs, nu=_nu(har.nu), origin=origin),
    }
    for name, template in (("rv-har", RV_HAR), ("iv-har", IV_HAR)):
        if name in rungs:
            rung = rungs[name]
            programs[name] = template.format(
                windows=tuple(rung.params["windows"]),
                coefs=coefficients(rung),
                nu=_nu(rung.nu),
                origin=origin,
                senses_rv=SENSES_RV,
            )
    if "ev-har" in rungs:
        rung = rungs["ev-har"]
        beta = (
            "{\n"
            + "".join(
                f"    {h}: ({', '.join(repr(float(v)) for v in rung.params['overlay'][str(h)]['beta'])}),\n"
                for h in sorted(map(int, rung.params["overlay"]))
            )
            + "}"
        )
        programs["ev-har"] = EV_HAR.format(
            windows=tuple(rung.params["windows"]),
            coefs=coefficients(rung),
            beta=beta,
            nu=_nu(rung.nu),
            origin=origin,
            senses_rv=SENSES_RV,
        )
    return programs


def _between(text: str, start: str, end: str) -> str:
    return text[text.index(start) : text.index(end)]


# EV-GBM's program shares EV-NET's code for EV-HAR and the inputs, character for character.
EV_GBM = (
    """\
import math
import numpy as np

WINDOWS = {windows!r}
HOURS_PER_YEAR = 8760
RELEASES = ("fomc", "cpi", "nfp")
FLOOR = 0.001  # final forecasts mix in the uniform distribution at this weight
ASSETS = {assets!r}
# EV-HAR, exactly as in its own program: per asset and horizon its intercept, one coefficient per window of 5-minute
# realized variance and one for the DVOL-implied variance (COEFS); its overlay's coefficients (BETA); its tails (NU).
# Then boosted trees per horizon that correct EV-HAR's log-probabilities, in the program's parameter file: for horizon
# h, ctx.params holds "h<h>.root", ".feature", ".threshold", ".left", ".right" and ".value". All fitted on {origin}.
COEFS = {{
{coefs}
}}
BETA = {beta}
NU = {nu}


"""
    + _between(EV_NET, "def _filled_of(", "def _tensors(")
    + _between(EV_NET, "def _net_inputs(", "def forecast(")
    + '''\
def _walk(params, h, x):
    """Per bin, the sum of the leaves that inputs x reach in horizon h's trees. A node sends x left when
    x[feature] <= threshold; a child c >= 0 is a node, c < 0 is leaf ~c; tree t adds its leaf to bin t % 10."""
    feature, threshold = params[f"h{{h}}.feature"], params[f"h{{h}}.threshold"]
    left, right, value = params[f"h{{h}}.left"], params[f"h{{h}}.right"], params[f"h{{h}}.value"]
    node = np.array(params[f"h{{h}}.root"])
    inner = node >= 0
    while inner.any():
        k = np.where(inner, node, 0)
        node = np.where(inner, np.where(x[feature[k]] <= threshold[k], left[k], right[k]), node)
        inner = node >= 0
    return value[~node].reshape(-1, 10).sum(axis=0)


def _ev_har(ctx):
    """EV-HAR's standard deviation and its final distribution, as in its own program."""
    filled = _filled(ctx)
    means = [float(np.mean(filled[-w:])) for w in WINDOWS]
    implied = _implied(ctx)
    if not math.isfinite(implied):
        implied = means[3]
    c = COEFS[ctx.asset][ctx.horizon]
    var = c[0] + (sum(k * m for k, m in zip(c[1:5], means)) + c[5] * implied) * ctx.horizon
    floor = 0.1 * means[2] * ctx.horizon
    scale = math.exp(0.5 * sum(b * x for b, x in zip(BETA[ctx.horizon], _inputs(ctx))))
    sd = math.sqrt(max(var, floor)) * scale
    base = np.asarray(ctx.lib.student_t_probs(sd, NU[ctx.horizon], ctx.edges), dtype=np.float64)
    return sd, (1 - FLOOR) * base / base.sum() + FLOOR / len(base)


def forecast(ctx):
    """EV-GBM: EV-HAR's distribution, corrected by boosted trees trained on the binned log score."""
    sd, base = _ev_har(ctx)
    x = np.array([*_net_inputs(ctx, sd), ASSETS.index(ctx.asset), ctx.hour, ctx.weekday], dtype=np.float64)
    logits = np.log(base) + _walk(ctx.params, ctx.horizon, x)
    p = np.exp(logits - logits.max())
    return p / p.sum()
'''
)


def _coefs(rung: Baseline, assets: Sequence[str]) -> str:
    return "\n".join(
        f"    {asset!r}: {{\n"
        + "".join(
            f"        {h}: ({', '.join(repr(float(c)) for c in by_h[str(h)])}),\n" for h in sorted(map(int, by_h))
        )
        + "    },"
        for asset, by_h in zip(assets, rung.params["per_asset"], strict=True)
    )


def _beta(rung: Baseline) -> str:
    return (
        "{\n"
        + "".join(
            f"    {h}: ({', '.join(repr(float(v)) for v in rung.params['overlay'][str(h)]['beta'])}),\n"
            for h in sorted(map(int, rung.params["overlay"]))
        )
        + "}"
    )


def gbm_program(ev: Baseline, assets: Sequence[str], origin: str) -> str:
    """Source code for EV-GBM from a fitted EV-HAR. Its trees are not in the source: they go in the program's parameter
    file (``Gbm.arrays()``, encoded by ``evolve.params``)."""
    return EV_GBM.format(
        windows=tuple(ev.params["windows"]),
        assets=tuple(assets),
        coefs=_coefs(ev, assets),
        beta=_beta(ev),
        nu=_nu(ev.nu),
        origin=origin,
    )
