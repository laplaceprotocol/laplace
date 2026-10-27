import math
import numpy as np

# LAMBDA: the decay. NU: Student-t degrees of freedom per horizon, chosen by binned log score.
# Both fitted on evolve-poc2's search window (2025-02-01 to 2026-06-01).
LAMBDA = 0.98
NU = {1: 4.0, 4: 4.0, 24: 5.0}


def forecast(ctx):
    """EWMA (RiskMetrics) variance of hourly returns, with Student-t tails."""
    r = np.diff(np.log(ctx.close))
    r2 = r * r
    var = float(np.mean(r2[:168]))
    for x in r2:
        var = LAMBDA * var + (1 - LAMBDA) * float(x)
    return ctx.lib.student_t_probs(math.sqrt(var * ctx.horizon), NU[ctx.horizon], ctx.edges)
