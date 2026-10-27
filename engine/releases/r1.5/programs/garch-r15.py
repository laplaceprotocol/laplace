import math
import numpy as np

# Per asset: omega, alpha, beta, fitted by maximum likelihood with Student-t innovations.
# NU: Student-t degrees of freedom per horizon, chosen by binned log score.
# Both fitted on evolve-refit2610's panel from 2025-02-01 to 2026-09-27 00:00 UTC.
PARAMS = {
    'BTC': (7.535320009992175e-07, 0.15869428945146846, 0.806615029815925),
    'ETH': (1.006332907387597e-06, 0.1009239383194068, 0.8791904863031379),
    'SOL': (1.0794780249106723e-06, 0.08910834374804427, 0.8940585709313593),
    'XRP': (1.3819284087564385e-06, 0.14105157922907563, 0.8375116486010762),
    'DOGE': (1.3586321681622428e-06, 0.10223092002934783, 0.8795596688726725),
    'AVAX': (2.634998078582333e-06, 0.11070560344885733, 0.8557065312623731),
    'LINK': (1.637031327236419e-06, 0.09517921325540145, 0.8828803558772418),
}
NU = {1: 4.0, 4: 4.0, 24: 5.0}


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
