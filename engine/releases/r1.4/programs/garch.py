import math
import numpy as np

# Per asset: omega, alpha, beta, fitted by maximum likelihood with Student-t innovations.
# NU: Student-t degrees of freedom per horizon, chosen by binned log score.
# Both fitted on evolve-poc2's search window (2025-02-01 to 2026-06-01).
PARAMS = {
    'BTC': (8.461539710317239e-07, 0.16554423886441302, 0.7965661693125136),
    'ETH': (1.1992189485075887e-06, 0.09874741671022073, 0.8793466160742185),
    'SOL': (2.0120854999456507e-06, 0.11078567408357165, 0.8601815185176085),
    'XRP': (1.902899536843276e-06, 0.15653211677445578, 0.8153210655599872),
    'DOGE': (2.3621185358157717e-06, 0.11221269776785628, 0.8586152910149449),
    'AVAX': (3.3369662108076946e-06, 0.12196064645140212, 0.8372694676198118),
    'LINK': (1.7965059199149733e-06, 0.09789283122415514, 0.8798055200860475),
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
