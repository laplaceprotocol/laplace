"""The baseline ladder as forecasting programs, with parameters frozen from a research fit.

Baselines run exactly like evolved programs: the same sandbox, the same past-only window, and their source is public.
Each program reproduces its research implementation (``laplace_research.evolve.baselines``) on the 720-hour window a
program sees. Two details differ, both far below a basis point: recursions start from the window's own mean squared
return rather than the full history's, and HAR's 720-hour mean covers the window's 719 returns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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


def _nu(nu: Mapping[int, float]) -> str:
    return "{" + ", ".join(f"{h}: {float(v)!r}" for h, v in sorted(nu.items())) + "}"


def ladder_programs(ladder: Sequence[Baseline], assets: Sequence[str], origin: str) -> dict[str, str]:
    """Source code for naive, EWMA, GARCH and HAR (and RV-HAR and IV-HAR when the ladder has them) from a fitted
    research ladder, keyed by program name."""
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
    return programs
